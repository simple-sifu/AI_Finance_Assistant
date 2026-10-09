# Deploying the AI Finance Tutor on AWS EC2

The app runs as one Docker container on one EC2 `t3.small` (2 GB RAM). It is served at `http://<elastic-ip>` behind a shared password.

- **Setup time:** about 30 minutes the first time.
- **Who builds:** everything is built on the server, so nothing needs to be installed on your computer.
- **Cost:** on a new AWS free-plan account (created on or after 2025-07-15), `t3.small` is free-tier eligible. The instance, a 20 GB disk and a public IPv4 address cost roughly $20 a month, taken from your $100–$200 of credits.
- **When the free plan ends:** after 6 months, or when the credits run out, whichever comes first.
- **Grading done?** Shut it down (step 9).

## 1. Set a spending alert (2 minutes)

1. In the AWS console, open **Billing and Cost Management → Budgets → Create budget**.
2. Choose **Use a template → Zero spend budget** (or a monthly cost budget of, say, $5) and enter your email.

You'll then get an email if anything is ever charged beyond the free credits.

## 2. Launch the instance

Open **EC2 → Instances → Launch instances**. Pick the region closest to your grader (e.g. `us-east-1`), then fill in:

| Setting | Value |
|---|---|
| Name | `finance-tutor` |
| Application and OS image | **Ubuntu Server 24.04 LTS** (64-bit x86) |
| Instance type | **t3.small** (shows "Free tier eligible") |
| Key pair | **Create new key pair**: name `finance-tutor`, type RSA, format `.pem`. Your browser downloads `finance-tutor.pem`; keep it. |
| Network settings | **Create security group** with: **Allow SSH traffic from: My IP**, and **Allow HTTP traffic from the internet**. Leave HTTPS unchecked. |
| Configure storage | **20** GiB, **gp3** |

Click **Launch instance**.

## 3. Give it a fixed address (Elastic IP)

Without an Elastic IP, the public address changes every time the instance stops.

1. Open **EC2 → Network & Security → Elastic IPs → Allocate Elastic IP address → Allocate**.
2. Select the new address, then **Actions → Associate Elastic IP address**.
3. Choose the `finance-tutor` instance and click **Associate**.

Write the address down; it's your app's URL from now on: `http://<elastic-ip>`.

## 4. Connect to the server

Either option works.

- **In the browser (easiest):**
  1. Open **EC2 → Instances**, select `finance-tutor`, then **Connect → EC2 Instance Connect → Connect**.
  2. If that fails because SSH is limited to "My IP", edit the security group's SSH rule to allow the EC2 Instance Connect range for your region. Or use the Terminal option below.
- **From your Mac's Terminal:**

  ```bash
  chmod 400 ~/Downloads/finance-tutor.pem
  ssh -i ~/Downloads/finance-tutor.pem ubuntu@<elastic-ip>
  ```

## 5. First run: install, then fill in your keys

On the server:

```bash
curl -fsSL https://raw.githubusercontent.com/simple-sifu/AI_Finance_Assistant/main/deploy/setup-server.sh -o setup-server.sh
bash setup-server.sh
```

The script is downloaded from `main`, so merge the deployment PR first. To try an unmerged branch instead, replace `main` with the branch name in the URL, and run the script as `BRANCH=<branch-name> bash …` every time.

The first run:
- adds a 2 GB swap file;
- installs Docker and git;
- clones the repo;
- creates `~/finance-assistant.env`, then stops.

Open that file and fill in the existing lines; don't paste a second copy below them:

```bash
nano ~/finance-assistant.env
```

```ini
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
MARKET_DATA_MODE=live
QUOTE_CACHE_TTL_SECONDS=1800
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-...
APP_PASSWORD=pick-a-password-for-the-grader
```

Save with Ctrl+O, Enter, then exit with Ctrl+X.

- Write each value without quotes and without a trailing `# comment`; Docker would pass either through as part of the value.
- The script refuses to start the app while `APP_PASSWORD` is empty.
- This file lives only on the server. Never commit it.

## 6. Build and start

```bash
bash ~/AI_Finance_Assistant/deploy/setup-server.sh
```

From now on, always run this copy from the repo, so fixes to the script arrive with each update.

The first build takes about 5–10 minutes: it downloads CPU-only PyTorch, builds the knowledge index and checks it offline. The build log should include:

```
Indexed … articles as … chunks into /app/knowledge_base/index
Offline index check ok: … articles; top hit 'Compound interest'
```

The script ends with `App is up: http://<elastic-ip>`.

## 7. Check it

1. **Health:** on the server, run `curl -s localhost/_stcore/health`. It should print `ok`.
2. **Password:** open `http://<elastic-ip>` in a browser. Only a password field shows. A wrong password says "Incorrect password."; the right one opens the five tabs.
3. **One question per agent:**

   | Tab | Question |
   |---|---|
   | Chat | What is compound interest? |
   | Portfolio | Upload `tests/data/sample_holdings.csv` from the repo, then ask: How diversified is this? |
   | Markets (market box) | How is AAPL doing? |
   | Markets (news box) | Latest Federal Reserve news |
   | Goals | Target 50000, 10 years, 6%, 5000 saved |
   | Knowledge | How is a Roth IRA different from a traditional IRA? |
   | Any tab | What should I buy? (expect the "I can't tell you what to buy…" redirect) |

   Every answer should end with the disclaimer. The cited answers (Knowledge, Chat, News) should list sources as links.
4. **Quota-exhausted demo (the success signal):**
   1. Set `MARKET_DATA_MODE=mock` in `~/finance-assistant.env` and run `bash ~/AI_Finance_Assistant/deploy/setup-server.sh` again. The image is cached, so this is quick.
   2. Market answers now label each ticker "(demo data, not a real-time price)", and every other agent works as before.
   3. Set it back to `live` afterwards. With a real exhausted key, the app falls back to cached quotes, then mock data, the same way.
5. **Offline model (optional):** this shows the image never needs Hugging Face.

   ```bash
   sudo docker run --rm --network none finance-assistant python /tmp/check_index.py
   ```

## 8. Update after merging to `main`

```bash
bash ~/AI_Finance_Assistant/deploy/setup-server.sh
```

It pulls the latest `main`, rebuilds and swaps the container. The app is down for a few seconds.

**Useful commands:**

```bash
sudo docker logs --tail 100 finance-assistant   # app logs
sudo docker ps                                  # is it running / healthy?
sudo docker restart finance-assistant           # restart without rebuilding
```

The container restarts by itself after a crash or a reboot (`--restart unless-stopped`).

## Security trade-offs (plain HTTP)

- The password and every question travel unencrypted over HTTP. Use a password made only for this demo, never one you use anywhere else.
- Nothing limits how many passwords someone can try. Keep the password long, and shut the instance down after grading.
- Each browser tab asks for the password once. A page refresh or a new tab asks again, because Streamlit keeps sign-in per session.

## 9. Shut it down after grading

- **Pause:** **EC2 → Instances → finance-tutor → Instance state → Stop**.
  - Compute charges stop.
  - The disk (about $1.60/month) and the Elastic IP (about $3.60/month) still use credits.
  - Start it again later; the container comes back on its own.
- **Remove everything:**
  1. **Instance state → Terminate** the instance.
  2. **Elastic IPs → Release** the address.
  3. Optionally delete the `finance-tutor` key pair and security group.
