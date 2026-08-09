# Running the ingest on AWS Fargate (optimized)

One-shot containerized run of `ingest -> pack --strict -> upload` on Fargate.
Compute is trivial; this is a bandwidth play, so the design optimizes for
**cheap, fail-fast, and no wasted hours**. See [../RUNBOOK.md](../RUNBOOK.md) for
the EC2 alternative and the bandwidth background.

## Why these choices
- **Speed gate as a separate ~1-min task first** — Fargate reaches GRID over
  datacenter egress, but GRID's *server-side* ceiling is unknown. Run the `gate`
  task before the big one; if it can't beat ~5 MB/s, stop (a VM won't help).
- **Secrets in SSM Parameter Store** — injected as env vars by the task; never in
  the image, the task def, or git.
- **Public subnet + public IP** — free internet egress; avoids a NAT gateway
  (~$0.045/hr + $0.045/GB) just to reach GRID/HF.
- **1 vCPU / 2 GB / 30 GB ephemeral** — process_game is ~2.4s and raw feeds are
  deleted per game, so peak disk is ~8 GB (processed + pack). Smaller works;
  this is the safe sweet spot.
- **On-demand, run-to-completion** by default. Fargate's disk is ephemeral, so an
  interrupted task loses the resumable state. For Spot/cheaper + resumable, see
  **EFS option** at the end.

Prereqs: AWS CLI v2 configured, Docker, and a VPC with a **public subnet**. Set:
```bash
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REGION=us-east-1
export SUBNET_ID=subnet-xxxxxxxx        # a PUBLIC subnet (route to IGW)
export SG_ID=sg-xxxxxxxx                # SG allowing outbound (default SG is fine)
```

## 1. Store secrets in SSM (SecureString)
```bash
aws ssm put-parameter --name /lol-winprob/GRID_API_KEY --type SecureString --value 'YOUR_GRID_KEY' --region $REGION
aws ssm put-parameter --name /lol-winprob/HF_TOKEN     --type SecureString --value 'YOUR_HF_TOKEN'  --region $REGION
```

## 2. IAM execution role
```bash
aws iam create-role --role-name lolWinprobExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# ECR pull + CloudWatch Logs
aws iam attach-role-policy --role-name lolWinprobExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# read the two SSM secrets (+ decrypt with the aws/ssm managed key)
aws iam put-role-policy --role-name lolWinprobExecutionRole --policy-name ssmRead \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"ssm:GetParameters\"],\"Resource\":\"arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/lol-winprob/*\"},{\"Effect\":\"Allow\",\"Action\":[\"kms:Decrypt\"],\"Resource\":\"arn:aws:kms:${REGION}:${ACCOUNT_ID}:alias/aws/ssm\"}]}"
```

## 3. Build + push the image to ECR
```bash
aws ecr create-repository --repository-name lol-winprob --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# build from the repo root using the deploy/ Dockerfile
docker build -f deploy/Dockerfile -t lol-winprob .
docker tag  lol-winprob:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/lol-winprob:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/lol-winprob:latest
```

## 4. Register the task definition
Edit `task-definition.json` — replace `<ACCOUNT_ID>`/`<REGION>` (and confirm
`HF_DATASET_REPO`). Then:
```bash
aws ecs register-task-definition --cli-input-json file://deploy/fargate/task-definition.json --region $REGION
aws ecs create-cluster --cluster-name lol-winprob --region $REGION
```

## 5. Speed gate first (cheap pre-flight)
Override the command to `gate`, then read the log:
```bash
NET="awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"

aws ecs run-task --cluster lol-winprob --launch-type FARGATE \
  --task-definition lol-winprob-ingest --region $REGION \
  --network-configuration "$NET" \
  --overrides '{"containerOverrides":[{"name":"ingest","command":["gate"]}]}'
```
Watch it (CloudWatch group `/ecs/lol-winprob-ingest`):
```bash
aws logs tail /ecs/lol-winprob-ingest --follow --region $REGION
```
- **≥ 5 MB/s** → proceed to step 6.
- **~0.6 MB/s** → GRID is the cap; stop and reconsider scope.

## 6. Full run
```bash
aws ecs run-task --cluster lol-winprob --launch-type FARGATE \
  --task-definition lol-winprob-ingest --region $REGION \
  --network-configuration "$NET" \
  --overrides '{"containerOverrides":[{"name":"ingest","command":["all"]}]}'
```
`all` = gate → ingest → pack --strict → upload. It streams progress to the same
log group and, on success, prints the HF dataset URL and the commit revision to
pin. The task exits when the upload finishes.

## 7. Cleanup
```bash
aws ecr delete-repository --repository-name lol-winprob --force --region $REGION
aws ssm delete-parameter --name /lol-winprob/GRID_API_KEY --region $REGION
aws ssm delete-parameter --name /lol-winprob/HF_TOKEN --region $REGION
aws ecs delete-cluster --cluster lol-winprob --region $REGION
# (optional) delete the IAM role + log group
```

## Cost
~1 vCPU/2 GB Fargate ≈ **$0.05/hr**; +10 GB ephemeral is negligible. Downloads
IN are free; the ~8 GB pack upload OUT to HF is ~$0.72. A full multi-hour run is
a **couple of dollars**. The gate task is a fraction of a cent.

## EFS option (Spot + resumable)
To use **Fargate Spot** (~70% cheaper) safely, persist `/app/data` on EFS so an
interrupted task resumes instead of restarting:
1. Create an EFS file system + a mount target in your subnet (SG allowing NFS 2049
   from `$SG_ID`).
2. Add to the task def: a `volumes` entry (`efsVolumeConfiguration`) and a
   `mountPoints` entry mapping it to `/app/data`.
3. Run with `--capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1`
   instead of `--launch-type FARGATE`.
Now `pipeline_state.json` and processed output survive interruptions; rerun the
task and it continues from the saved state. Skip this if the on-demand run
finishes in one sitting — EFS adds moving parts (mount targets, NFS SG rules).
