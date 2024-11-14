import os
from github import Github

# Replace with your GitHub personal access token
token = GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
repo_name = "Microsaofts/KeyAuth-Login-Script-Loader"

# Initialize the GitHub object
g = Github(token)

# Get the repository object
repo = g.get_repo(repo_name)

# Retrieve the contents of the root directory
contents = repo.get_contents("")

# Recursive function to list all files
def list_files(content_list):
    files = []
    for content in content_list:
        if content.type != "dir":
            # If it's a file, add its path
            files.append(content.path)
    return files

# List all files in the repository
all_files = list_files(contents)

# Print the file paths
for file in all_files:
    print(file)