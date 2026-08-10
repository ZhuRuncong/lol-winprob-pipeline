# Running the ingest on Fargate — AWS Console (UI) walkthrough

Same result as [README.md](README.md), done by clicking through the console.
Pick a region first (top-right) and stay in it the whole time — everything below
must live in one region. This guide uses **us-east-1** as the example.

One step still needs a terminal: **building and pushing the container image**
(step 3). There's no pure-console way to build a Docker image from a Dockerfile;
the ECR console hands you the exact commands to run. Everything else is UI.

---

## 1. Store the two secrets — Systems Manager
Console → **Systems Manager** → **Parameter Store** → **Create parameter**. Do
this twice:

| Name | Type | Value |
|------|------|-------|
| `/lol-winprob/GRID_API_KEY` | SecureString | your GRID key |
| `/lol-winprob/HF_TOKEN` | SecureString | your HF write token |

Leave the KMS key as the default (`alias/aws/ssm`). Create.

---

## 2. Execution role — IAM
Console → **IAM** → **Roles** → **Create role**.
1. Trusted entity: **AWS service** → use case **Elastic Container Service** →
   **Elastic Container Service Task** → Next.
2. Attach policy **AmazonECSTaskExecutionRolePolicy** → Next.
3. Name it `lolWinprobExecutionRole` → Create role.
4. Open the new role → **Add permissions** → **Create inline policy** → **JSON**,
   paste (replace `<REGION>`/`<ACCOUNT_ID>` — your account id is top-right under
   your name):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       { "Effect": "Allow", "Action": ["ssm:GetParameters"],
         "Resource": "arn:aws:ssm:<REGION>:<ACCOUNT_ID>:parameter/lol-winprob/*" },
       { "Effect": "Allow", "Action": ["kms:Decrypt"],
         "Resource": "arn:aws:kms:<REGION>:<ACCOUNT_ID>:alias/aws/ssm" }
     ]
   }
   ```
   Name it `ssmRead` → Create. (Without this, the task can't read the secrets.)

---

## 3. Container image — ECR (needs a terminal for the build)
Console → **ECR** → **Create repository** → name `lol-winprob` → Create.
Open the repo → **View push commands** → run those 4 commands **from the repo
root on a machine with Docker**, but change the 3rd one so it uses this
Dockerfile:
```
docker build -f deploy/Dockerfile -t lol-winprob .
```
(the other three — login, tag, push — copy verbatim from the dialog). When it
finishes, the image shows up under the repo with tag `latest`.

> No Docker locally? Create an **AWS CodeBuild** project pointing at the GitHub
> repo with a buildspec that runs the same build+push — but if you have Docker,
> the push-commands route is far quicker.

---

## 4. Cluster — ECS
Console → **ECS** → **Clusters** → **Create cluster**. Name `lol-winprob`,
Infrastructure **AWS Fargate** (default), Create.

---

## 5. Task definition — ECS
ECS → **Task definitions** → **Create new task definition** (the form, not JSON).
- **Family:** `lol-winprob-ingest`
- **Launch type:** AWS Fargate · OS **Linux/X86_64**
- **CPU:** 1 vCPU · **Memory:** 2 GB
- **Task role:** none · **Task execution role:** `lolWinprobExecutionRole`
- **Ephemeral storage:** set to **30 GB**
- **Container 1:**
  - Name `ingest`
  - Image URI: `‹ACCOUNT›.dkr.ecr.‹REGION›.amazonaws.com/lol-winprob:latest`
    (copy from the ECR repo page)
  - Essential: yes · remove any default **port mapping** (not needed)
  - **Environment variables** (add three, value type **Value**):
    - `HF_DATASET_REPO` = `zauberine/lol-winprob`
    - `HF_PRIVATE` = `0`
    - `START_AFTER` = `2026-01-01`
  - **Secrets / environment (value type `ValueFrom`)** — add two, pasting the
    SSM **parameter ARNs** from step 1:
    - `GRID_API_KEY` → `arn:aws:ssm:‹REGION›:‹ACCOUNT›:parameter/lol-winprob/GRID_API_KEY`
    - `HF_TOKEN` → `arn:aws:ssm:‹REGION›:‹ACCOUNT›:parameter/lol-winprob/HF_TOKEN`
  - **Logging:** leave **Use log collection** on (Amazon CloudWatch) — it creates
    log group `/ecs/lol-winprob-ingest`.
- Create.

---

## 6. Speed gate first (cheap pre-flight)
ECS → Clusters → **lol-winprob** → **Tasks** tab → **Run new task**.
- **Launch type:** Fargate · **Task definition:** `lol-winprob-ingest` (latest)
- **Networking:**
  - VPC: your default (or chosen) VPC
  - **Subnets:** pick a **public** subnet
  - Security group: default is fine (needs outbound; no inbound)
  - **Public IP:** **Turned on**  ← required for internet egress without a NAT
- Expand **Container overrides** → **ingest** → **Command** → type: `gate`
- **Create** (run).

Watch it: the task's **Logs** tab (or CloudWatch → Log groups →
`/ecs/lol-winprob-ingest`). It prints MB/s:
- **≥ 5 MB/s** → go to step 7.
- **~0.6 MB/s** → GRID itself is the cap; stop, reconsider scope.

The task exits on its own after the measurement.

---

## 7. Full run
Repeat **Run new task** exactly as step 6, but set **Command** override to `all`
(instead of `gate`). This runs gate → ingest → pack `--strict` → upload.

Monitor the same log stream. On success it prints the dataset URL
(`https://huggingface.co/datasets/zauberine/lol-winprob`) and a **commit
revision** — copy that to pin in training. The task stops when the upload
finishes.

Progress is per-series; if a task dies mid-run, just **Run new task** again with
`all` — but note Fargate's disk is ephemeral, so it restarts from the beginning
(the fetched list is rebuilt automatically). For resumability, use the EFS option
in [README.md](README.md).

---

## 8. Cleanup
- **ECS** → delete the cluster (and the task definition if you want).
- **ECR** → delete the `lol-winprob` repository.
- **Systems Manager** → delete both parameters.
- **CloudWatch** → delete log group `/ecs/lol-winprob-ingest` (optional).
- **IAM** → delete `lolWinprobExecutionRole` (optional).

## Cost
1 vCPU/2 GB Fargate ≈ $0.05/hr; the gate task is a fraction of a cent; the full
run is a couple of dollars (mostly instance-hours + ~$0.72 pack upload). Nothing
bills after you stop the task and delete the repo/cluster.
