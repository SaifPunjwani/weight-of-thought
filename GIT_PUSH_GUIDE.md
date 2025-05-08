# Git Push Guide for Weight-of-Thought Repository

Use the following step-by-step guide to push your codebase to GitHub.

## Prerequisites
- Git installed on your computer
- GitHub account (username: SaifPunjwani)
- GitHub repository created: https://github.com/SaifPunjwani/weight-of-thought
- Git configured with your credentials

## Step 1: Configure Git

First, configure Git with your credentials:

```bash
git config user.name "Saif Punjwani"
git config user.email "saifpunjwani1230@gmail.com"
```

## Step 2: Initialize Git Repository (if not already done)

```bash
git init
```

## Step 3: Set the Correct Remote Repository

Remove any existing remote repositories and add the correct one:

```bash
# Remove existing remote if it exists
git remote remove origin

# Add the correct remote
git remote add origin git@github.com:SaifPunjwani/weight-of-thought.git

# Verify remote
git remote -v
```

## Step 4: Add and Commit Your Files

Add all files to staging and commit:

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: Weight-of-Thought Reasoning Framework"
```

## Step 5: Handle the Existing Remote Content

Since the repository already has content (it was rejecting your push), you have two options:

### Option A: Force Push (recommended if you want to replace everything)

```bash
git push -f origin main
```

### Option B: Pull First, Then Push

```bash
git pull --rebase origin main
git push origin main
```

## Step 6: Verify Repository Content

After pushing, visit your GitHub repository to make sure all files were uploaded correctly:
https://github.com/SaifPunjwani/weight-of-thought

## Step 7: Push the REALM Paper Images

Make sure the folder "REALM Paper (arXiv)" is included in your push, as it contains the important images for your README.

## Troubleshooting

### Issue: SSH Key Authentication

If you're using SSH and encounter authentication issues:

```bash
# Check if your SSH key is registered with the SSH agent
ssh-add -l

# If not, add it
ssh-add ~/.ssh/id_rsa  # or your key path

# Test SSH connection
ssh -T git@github.com
```

### Issue: Large Files

If you have files larger than 100MB, GitHub will reject them:

```bash
# Move large files to an ignored directory
mkdir -p large_files
mv [large_file_path] large_files/

# Add the directory to .gitignore
echo "large_files/" >> .gitignore

# Then add and commit again
git add .
git commit -m "Remove large files"
```

### Issue: HTTPS Authentication

If using HTTPS instead of SSH:

```bash
git remote set-url origin https://github.com/SaifPunjwani/weight-of-thought.git
```

When pushing, you'll be prompted for your GitHub username and a personal access token (not your GitHub password).

## Need More Help?

For more advanced Git issues, consult the [GitHub documentation](https://docs.github.com/en/get-started/using-git/about-git) or seek help from the GitHub community.