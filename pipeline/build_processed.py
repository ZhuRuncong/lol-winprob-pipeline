"""Build processed per-second training data from Riot esports live-event JSONL feeds.

Input : data/raw/<matchId>/events_<matchId>_<gameN>_riot.jsonl
Output: data/processed/<matchId>_<gameN>/{meta.json, sequence.npz}
        data/processed/features.json  (ordered feature-name lists)
        data/processed/norm.json      (per-feature mean/std, written by --all)

Usage:
  python -m pipeline.build_processed --one 2913131_1
  python -m pipeline.build_processed --all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

MAP_SIZE = 15000.0  # Summoner's Rift x/z extent
PLATES_FALL_S = 14 * 60  # plates disappear at 14:00
PLATES_PER_LANE = 5
INHIB_RESPAWN_S = 300
BARON_BUFF_S = 180
ELDER_BUFF_S = 150
WARD_KILL_MATCH_DIST = 250.0  # max units between ward_killed pos and a candidate ward

LANES = ("top", "mid", "bot")
DRAGON_TYPES = ("fire", "water", "earth", "air", "hextech", "chemtech")
WARD_TYPE_IDS = {"yellowTrinket": 0, "blueTrinket": 1, "sight": 2, "control": 3}
STEALTH_WARD_TYPES = {"yellowTrinket", "sight"}  # untracked expiry -> simulated
STEALTH_CAP_PER_PLAYER = 3

FEATURES_CHAMP = [
    "alive", "respawn_timer", "level", "xp", "current_gold", "total_gold",
    "hp", "hp_max", "hp_frac", "mana", "mana_max", "mana_frac",
    "cd_q", "cd_w", "cd_e", "cd_r", "lvl_q", "lvl_w", "lvl_e", "lvl_r",
    "cd_summ1", "cd_summ2", "pos_x", "pos_z", "shutdown_value", "cs",
]
FEATURES_CHAMP_KDA = ["kills", "deaths", "assists", "vision_score"]
FEATURES_TEAM = [
    "kills", "deaths", "assists", "total_gold",
    "towers_destroyed", "inhibs_destroyed",
    "turrets_alive_top", "turrets_alive_mid", "turrets_alive_bot", "nexus_turrets_alive",
    "plates_remaining_top", "plates_remaining_mid", "plates_remaining_bot",
    "inhib_down_timer_top", "inhib_down_timer_mid", "inhib_down_timer_bot",
    "dragons_fire", "dragons_water", "dragons_earth", "dragons_air",
    "dragons_hextech", "dragons_chemtech",
    "elder_active", "baron_active", "grubs_taken", "herald_taken",
]
FEATURES_GLOBAL = [
    "game_time_norm", "next_dragon_spawn_in", "next_baron_or_herald_spawn_in",
    "is_pre_plates", "dragon_up", "baron_or_herald_up",
]
WARD_COLUMNS = ["team", "type", "x", "z", "t_start", "t_end"]


def team_idx(team_id):
    return 0 if team_id == 100 else 1


def stealth_ward_duration(avg_level: float) -> float:
    """Regular (stealth) ward lifetime: 90 + 30/17 * (avg champion level - 1)."""
    return 90.0 + 30.0 / 17.0 * (avg_level - 1.0)


# ---------------------------------------------------------------- streaming parse

def parse_game(path: Path):
    """Single streaming pass. Returns (header, ticks, events) or None if unusable."""
    champ_select = None
    game_info = None
    game_end = None
    ticks = {}       # sec -> extracted tick payload (keep last per sec)
    ev = {k: [] for k in (
        "building_destroyed", "turret_plate_destroyed", "epic_monster_kill",
        "epic_monster_spawn", "queued_dragon_info", "queued_epic_monster_info",
        "ward_placed", "ward_killed",
    )}

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = d.get("rfc461Schema")
            if s == "stats_update":
                sec = int(round(d["gameTime"] / 1000.0))
                ticks[sec] = extract_tick(d)
            elif s in ev:
                ev[s].append(d)
            elif s == "champ_select":
                champ_select = d  # keep last (final rosters/bans)
            elif s == "game_info":
                game_info = d
            elif s == "game_end":
                game_end = d

    if game_end is None or game_info is None or not ticks:
        return None
    for lst in ev.values():
        lst.sort(key=lambda e: e["gameTime"])
    return {"champ_select": champ_select, "game_info": game_info, "game_end": game_end}, ticks, ev


def extract_tick(d):
    """Reduce one stats_update to per-champ / team-score arrays."""
    champ = np.zeros((10, len(FEATURES_CHAMP)), dtype=np.float32)
    kda = np.zeros((10, len(FEATURES_CHAMP_KDA)), dtype=np.float32)
    items = np.zeros((10, 7), dtype=np.int32)
    item_cd = np.zeros((10, 7), dtype=np.float32)

    for p in d["participants"]:
        i = p["participantID"] - 1
        if not 0 <= i < 10:
            continue
        st = {s["name"]: s["value"] for s in p.get("stats", [])}
        hp, hp_max = p.get("health", 0), p.get("healthMax", 0)
        mana, mana_max = p.get("primaryAbilityResource", 0), p.get("primaryAbilityResourceMax", 0)
        pos = p.get("position", {})
        champ[i] = [
            1.0 if p.get("alive") else 0.0,
            p.get("respawnTimer", 0),
            p.get("level", 0),
            p.get("XP", 0),
            p.get("currentGold", 0),
            p.get("totalGold", 0),
            hp, hp_max, (hp / hp_max) if hp_max else 0.0,
            mana, mana_max, (mana / mana_max) if mana_max else 0.0,
            p.get("ability1CooldownRemaining", 0),
            p.get("ability2CooldownRemaining", 0),
            p.get("ability3CooldownRemaining", 0),
            p.get("ultimateCooldownRemaining", 0),
            p.get("ability1Level", 0), p.get("ability2Level", 0),
            p.get("ability3Level", 0), p.get("ability4Level", 0),
            p.get("summonerSpell1CooldownRemaining", 0),
            p.get("summonerSpell2CooldownRemaining", 0),
            pos.get("x", 0) / MAP_SIZE, pos.get("z", 0) / MAP_SIZE,
            p.get("shutdownValue", 0),
            st.get("MINIONS_KILLED", 0) + st.get("NEUTRAL_MINIONS_KILLED", 0),
        ]
        kda[i] = [
            st.get("CHAMPIONS_KILLED", 0), st.get("NUM_DEATHS", 0),
            st.get("ASSISTS", 0), st.get("VISION_SCORE", 0),
        ]
        for slot, it in enumerate(p.get("items", [])[:7]):
            items[i, slot] = it.get("itemID", 0)
            item_cd[i, slot] = it.get("itemCooldown", 0)

    team_score = np.zeros((2, 6), dtype=np.float32)  # kills deaths assists gold towerK inhibK
    for t in d.get("teams", []):
        j = team_idx(t["teamID"])
        team_score[j] = [
            t.get("championsKills", 0), t.get("deaths", 0), t.get("assists", 0),
            t.get("totalGold", 0), t.get("towerKills", 0), t.get("inhibKills", 0),
        ]
    return {"champ": champ, "kda": kda, "items": items, "item_cd": item_cd, "team_score": team_score}


# ---------------------------------------------------------------- reducers

def build_team_and_global(T, ev):
    """Per-second team-state [T,2,26] (reducer part) and global [T,6] arrays."""
    X_team = np.zeros((T, 2, len(FEATURES_TEAM)), dtype=np.float32)
    X_global = np.zeros((T, len(FEATURES_GLOBAL)), dtype=np.float32)

    # --- structure loss timelines (owner team loses the structure) ---
    turret_falls = []   # (sec, team, lane) lane turrets (outer/inner/base)
    nexus_falls = []    # (sec, team)
    inhib_falls = []    # (sec, team, lane)
    for d in ev["building_destroyed"]:
        sec, tm = int(round(d["gameTime"] / 1000)), team_idx(d["teamID"])
        bt = d.get("buildingType")
        if bt == "turret":
            if d.get("turretTier") == "nexus":
                nexus_falls.append((sec, tm))
            else:
                turret_falls.append((sec, tm, d.get("lane")))
        elif bt == "inhibitor":
            inhib_falls.append((sec, tm, d.get("lane")))

    plate_falls = [(int(round(d["gameTime"] / 1000)), team_idx(d["teamID"]), d.get("lane"))
                   for d in ev["turret_plate_destroyed"]]

    # --- epic monster timelines (killerTeamID gains) ---
    dragon_kills, elder_kills, baron_kills, grub_kills, herald_kills = [], [], [], [], []
    dragon_alive = []   # (spawn_sec, kill_sec) windows for dragon_up
    for d in ev["epic_monster_kill"]:
        sec, tm = int(round(d["gameTime"] / 1000)), team_idx(d.get("killerTeamID", 100))
        mt = d.get("monsterType")
        if mt == "dragon":
            if d.get("dragonType") == "elder":
                elder_kills.append((sec, tm))
            else:
                dragon_kills.append((sec, tm, d.get("dragonType")))
        elif mt == "baron":
            baron_kills.append((sec, tm))
        elif mt == "VoidGrub":
            grub_kills.append((sec, tm))
        elif mt == "riftHerald":
            herald_kills.append((sec, tm))

    dragon_spawns = sorted(int(round(d["gameTime"] / 1000)) for d in ev["epic_monster_spawn"]
                           if d.get("monsterType") == "dragon")
    all_dragon_deaths = sorted(s for s, _, _ in dragon_kills) + sorted(s for s, _ in elder_kills)
    all_dragon_deaths.sort()
    for sp in dragon_spawns:
        kill = next((k for k in all_dragon_deaths if k >= sp), T)
        dragon_alive.append((sp, kill))

    # queued spawn info: latest announcement wins (spawn times are in SECONDS)
    dragon_queue = [(int(round(d["gameTime"] / 1000)), d.get("nextDragonSpawnTime", 0))
                    for d in ev["queued_dragon_info"]]
    epic_queue = [(int(round(d["gameTime"] / 1000)), d.get("monsterName"), d.get("spawnTime", 0))
                  for d in ev["queued_epic_monster_info"]]

    for t in range(T):
        for tm in (0, 1):
            f = {}
            # own structures alive
            for lane in LANES:
                fallen = sum(1 for s, m, ln in turret_falls if m == tm and ln == lane and s <= t)
                f[f"turrets_alive_{lane}"] = 3 - fallen
                pl = sum(1 for s, m, ln in plate_falls if m == tm and ln == lane and s <= t)
                f[f"plates_remaining_{lane}"] = 0 if t >= PLATES_FALL_S else max(0, PLATES_PER_LANE - pl)
                down = [s for s, m, ln in inhib_falls if m == tm and ln == lane and s <= t]
                f[f"inhib_down_timer_{lane}"] = max(0, down[-1] + INHIB_RESPAWN_S - t) if down else 0
            f["nexus_turrets_alive"] = 2 - sum(1 for s, m in nexus_falls if m == tm and s <= t)
            # objectives taken by this team
            for dt in DRAGON_TYPES:
                f[f"dragons_{dt}"] = sum(1 for s, m, d_ in dragon_kills if m == tm and d_ == dt and s <= t)
            f["elder_active"] = float(any(m == tm and s <= t < s + ELDER_BUFF_S for s, m in elder_kills))
            f["baron_active"] = float(any(m == tm and s <= t < s + BARON_BUFF_S for s, m in baron_kills))
            f["grubs_taken"] = sum(1 for s, m in grub_kills if m == tm and s <= t)
            f["herald_taken"] = sum(1 for s, m in herald_kills if m == tm and s <= t)
            for name, val in f.items():
                X_team[t, tm, FEATURES_TEAM.index(name)] = val

        # --- global ---
        q = [sp for at, sp in dragon_queue if at <= t]
        next_dragon = max(0, q[-1] - t) if q else 0
        upcoming = []
        for name in ("Baron", "RiftHerald"):
            qq = [sp for at, nm, sp in epic_queue if at <= t and nm == name]
            if qq:
                upcoming.append(max(0, qq[-1] - t))
        dragon_up = float(any(a <= t < b for a, b in dragon_alive))
        baron_or_herald_up = float(any(u == 0 for u in upcoming)) if upcoming else 0.0
        X_global[t] = [
            t / 1800.0, next_dragon, min(upcoming) if upcoming else 0,
            float(t < PLATES_FALL_S), dragon_up, baron_or_herald_up,
        ]
    return X_team, X_global


def build_wards(T, ev, levels_by_sec):
    """Ward interval table [W,6]: (team, type, x, z, t_start, t_end)."""
    events = ([("place", d) for d in ev["ward_placed"]] +
              [("kill", d) for d in ev["ward_killed"]])
    events.sort(key=lambda e: e[1]["gameTime"])

    active = []   # dicts: placer, type_name, x, z, t_start, expiry
    closed = []   # (ward, t_end)

    def close(w, t_end):
        active.remove(w)
        closed.append((w, t_end))

    for kind, d in events:
        t = d["gameTime"] / 1000.0
        # expire stealth wards that timed out before this event
        for w in [w for w in active if w["expiry"] is not None and w["expiry"] <= t]:
            close(w, w["expiry"])
        pos = d.get("position", {})
        x, z = pos.get("x", 0), pos.get("z", 0)
        wt = d.get("wardType")
        if kind == "place":
            placer = d.get("placer", 0)
            expiry = None
            if wt in STEALTH_WARD_TYPES:
                lv = levels_by_sec[min(int(t), T - 1)]
                expiry = t + stealth_ward_duration(float(np.mean(lv)))
                # cap: max 3 stealth wards per player -> earliest is destroyed
                mine = sorted((w for w in active
                               if w["placer"] == placer and w["type_name"] in STEALTH_WARD_TYPES),
                              key=lambda w: w["t_start"])
                if len(mine) >= STEALTH_CAP_PER_PLAYER:
                    close(mine[0], t)
            active.append({"placer": placer, "type_name": wt, "x": x, "z": z,
                           "t_start": t, "expiry": expiry})
        else:  # kill: nearest active ward of same type within range
            cands = [w for w in active if w["type_name"] == wt]
            if cands:
                best = min(cands, key=lambda w: math.hypot(w["x"] - x, w["z"] - z))
                if math.hypot(best["x"] - x, best["z"] - z) <= WARD_KILL_MATCH_DIST:
                    close(best, t)

    for w in list(active):  # end of game: expiry or game end
        close(w, min(w["expiry"], float(T)) if w["expiry"] is not None else float(T))

    rows = np.zeros((len(closed), len(WARD_COLUMNS)), dtype=np.float32)
    for i, (w, t_end) in enumerate(closed):
        rows[i] = [
            0 if 1 <= w["placer"] <= 5 else 1,
            WARD_TYPE_IDS.get(w["type_name"], 0),
            w["x"] / MAP_SIZE, w["z"] / MAP_SIZE,
            w["t_start"], min(t_end, float(T)),
        ]
    return rows[rows[:, 4].argsort()]


# ---------------------------------------------------------------- meta

def build_meta(header, match_id, game_num, T):
    gi, cs, ge = header["game_info"], header["champ_select"], header["game_end"]
    winning = ge["winningTeam"]

    # champ_select participantIDs do NOT match in-game participantIDs -> join by puuid
    cs_by_puuid = {}
    if cs:
        for side in ("teamOne", "teamTwo"):
            for p in cs.get(side, []):
                cs_by_puuid[p.get("puuid")] = p

    participants = []
    for p in sorted(gi["participants"], key=lambda p: p["participantID"]):
        pid = p["participantID"]
        c = cs_by_puuid.get(p.get("puuid"), {})
        perks = (p.get("perks") or [{}])[0]
        participants.append({
            "participant_id": pid,
            "team": p["teamID"],
            "role": p.get("role"),
            "champion": p.get("championName"),
            "champion_id": c.get("championID"),
            "summoner_spells": [c.get("spell1"), c.get("spell2")],
            "runes": {
                "keystone": p.get("keystoneID"),
                "perk_ids": perks.get("perkIds", []),
                "style": perks.get("perkStyle"),
                "sub_style": perks.get("perkSubStyle"),
            },
            "player": p.get("summonerName"),
            "puuid": p.get("puuid"),
        })

    bans = [{"team": b["teamID"], "champion_id": b["championID"], "pick_turn": b["pickTurn"]}
            for b in (cs.get("bannedChampions", []) if cs else [])]

    return {
        "match_id": match_id, "game_num": game_num, "game_id": gi.get("gameID"),
        "patch": gi.get("gameVersion"), "platform": gi.get("platformID"),
        "duration_s": T,
        "winning_team": winning, "label": 1 if winning == 100 else 0,
        "teams": {"100": "blue", "200": "red"},
        "bans": bans, "participants": participants,
        "norm_version": "v1",
    }


# ---------------------------------------------------------------- game driver

def process_game(events_path: Path, match_id: int, game_num: int, out_root: Path):
    parsed = parse_game(events_path)
    if parsed is None:
        return None
    header, ticks, ev = parsed

    T = max(ticks) + 1
    n_c, n_k = len(FEATURES_CHAMP), len(FEATURES_CHAMP_KDA)
    X_champ = np.zeros((T, 10, n_c), dtype=np.float32)
    X_kda = np.zeros((T, 10, n_k), dtype=np.float32)
    X_items = np.zeros((T, 10, 7), dtype=np.int32)
    X_item_cd = np.zeros((T, 10, 7), dtype=np.float32)
    team_score = np.zeros((T, 2, 6), dtype=np.float32)

    prev = None
    for t in range(T):
        tick = ticks.get(t, prev)
        if tick is None:
            continue
        X_champ[t], X_kda[t] = tick["champ"], tick["kda"]
        X_items[t], X_item_cd[t] = tick["items"], tick["item_cd"]
        team_score[t] = tick["team_score"]
        prev = tick

    X_team, X_global = build_team_and_global(T, ev)
    X_team[:, :, :6] = team_score  # kills/deaths/assists/gold/towers/inhibs from stats ticks

    levels_by_sec = X_champ[:, :, FEATURES_CHAMP.index("level")]
    wards = build_wards(T, ev, levels_by_sec)

    meta = build_meta(header, match_id, game_num, T)

    out = out_root / f"{match_id}_{game_num}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "sequence.npz",
        t=np.arange(T, dtype=np.int32),
        X_champ=X_champ, X_champ_kda=X_kda,
        X_items=X_items, X_item_cd=X_item_cd,
        X_team=X_team, X_global=X_global,
        wards=wards,
        y=np.int32(meta["label"]),
    )
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {"dir": out, "T": T, "wards": len(wards), "label": meta["label"],
            "n_placed": len(ev["ward_placed"])}


# ---------------------------------------------------------------- dataset level

def write_features_json(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "features.json", "w", encoding="utf-8") as fh:
        json.dump({
            "X_champ": FEATURES_CHAMP, "X_champ_kda": FEATURES_CHAMP_KDA,
            "X_items": "itemID per slot (7 slots, 0 = empty)",
            "X_item_cd": "item cooldown remaining per slot (s)",
            "X_team": FEATURES_TEAM, "X_global": FEATURES_GLOBAL,
            "wards": WARD_COLUMNS,
            "ward_type_ids": WARD_TYPE_IDS,
            "y": "1 if blue (team 100) won",
        }, fh, indent=1)


def find_games():
    pat = re.compile(r"events_(\d+)_(\d+)_riot\.jsonl$")
    games = []
    for sub in sorted(RAW_DIR.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.glob("events_*_riot.jsonl")):
                m = pat.match(f.name)
                if m:
                    games.append((f, int(m.group(1)), int(m.group(2))))
    return games


def write_norm_json(out_root: Path, dirs):
    """Streaming per-feature mean/std across all processed games."""
    sums, sqs, cnt = {}, {}, {}
    keys = ("X_champ", "X_champ_kda", "X_item_cd", "X_team", "X_global")
    for d in dirs:
        with np.load(d / "sequence.npz") as z:
            for k in keys:
                a = z[k].astype(np.float64).reshape(-1, z[k].shape[-1])
                sums[k] = sums.get(k, 0) + a.sum(0)
                sqs[k] = sqs.get(k, 0) + (a ** 2).sum(0)
                cnt[k] = cnt.get(k, 0) + a.shape[0]
    norm = {}
    for k in keys:
        mean = sums[k] / cnt[k]
        var = np.maximum(sqs[k] / cnt[k] - mean ** 2, 1e-12)
        norm[k] = {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}
    with open(out_root / "norm.json", "w", encoding="utf-8") as fh:
        json.dump({"norm_version": "v1", "stats": norm}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help="process a single game, e.g. 2913131_1")
    ap.add_argument("--all", action="store_true", help="process every game under data/raw")
    args = ap.parse_args()
    write_features_json(OUT_DIR)

    if args.one:
        match_id, game_num = map(int, args.one.split("_"))
        f = RAW_DIR / str(match_id) / f"events_{match_id}_{game_num}_riot.jsonl"
        if not f.exists():
            sys.exit(f"not found: {f}")
        r = process_game(f, match_id, game_num, OUT_DIR)
        if r is None:
            sys.exit("skipped: no game_end / game_info / stats ticks")
        print(f"OK {r['dir']}  T={r['T']}s  wards={r['wards']}/{r['n_placed']} placed  label={r['label']}")
        return

    if args.all:
        done, skipped = [], []
        games = find_games()
        for i, (f, mid, gn) in enumerate(games, 1):
            try:
                r = process_game(f, mid, gn, OUT_DIR)
            except Exception as e:  # keep the batch going, report at the end
                skipped.append((f.name, f"error: {e}"))
                continue
            if r is None:
                skipped.append((f.name, "no game_end"))
            else:
                done.append(r["dir"])
            print(f"[{i}/{len(games)}] {f.name} -> {'ok' if r else 'skip'}", flush=True)
        if done:
            write_norm_json(OUT_DIR, done)
        print(f"\nprocessed {len(done)} games, skipped {len(skipped)}")
        for name, why in skipped:
            print(f"  skip {name}: {why}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
