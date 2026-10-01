# Colab setup

The repo is public, so Colab clones it without credentials. A token is needed
only to **push run results** back to GitHub (spec §7.5).

## One-time: create the GitHub token
1. GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → Generate new token.
2. **Repository access:** *Only select repositories* → `aavig23/PINN-elasticity-2`.
3. **Permissions:** Repository permissions → **Contents: Read and write**.
   Leave everything else at *No access*.
4. **Expiration:** short (e.g. 30 days). Renew when it expires.
5. Copy the token once; GitHub will not show it again.

## One-time: store it in Colab Secrets
1. In Colab, open the key icon (**Secrets**) in the left sidebar.
2. Add a secret named `GITHUB_TOKEN` with the token as its value.
3. Enable **Notebook access** for the runner notebook.

## Token rules
The token must never be hard-coded, typed into a cell, printed, logged,
written to any file, saved in notebook outputs, committed, or left in a git
remote URL or git config. The push script reads it from Colab Secrets at run
time and passes it to git only for the duration of the push.

## Git identity for Colab commits
Commits from Colab use your GitHub **noreply** email
(GitHub → Settings → Emails → "Keep my email addresses private" shows it,
in the form `<id>+<username>@users.noreply.github.com`).

*Notebook usage, Drive checkpoint backup and resume steps are added in M7.*
