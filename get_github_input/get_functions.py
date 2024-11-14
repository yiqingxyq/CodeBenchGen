import os
import json
import argparse
from tqdm import tqdm
import git

from check_repo_stats import check_repo_stats

from r2e.repo_builder.repo_args import RepoArgs
from r2e.repo_builder.extract_func_methods import build_functions_and_methods
from r2e.paths import REPOS_DIR, EXTRACTION_DIR

    
def clone_repo_from_url(full_repo_name: str):
    repo_url = f"https://github.com/{full_repo_name}.git"
    
    repo_username, repo_name = (
        repo_url.rstrip("/").removesuffix(".git").split("/")[-2:]
    )
    local_repo_clone_path = REPOS_DIR / f"{repo_username}___{repo_name}"

    if os.path.exists(local_repo_clone_path):
        # print(f"Repository {repo_url} already exists at {local_repo_clone_path}... skipping")
        return
    
    try:
        git.Repo.clone_from(f"{repo_url}", local_repo_clone_path)
        print(f"Successfully cloned repository {repo_url} to {local_repo_clone_path}")
    except:
        print(f"Error cloning from {repo_url}")
        

def extract_functions(disable_no_docstring=False, exp_id="temp"):
    repo_args = RepoArgs(
        overwrite_extracted=True, 
        disable_no_docstring=disable_no_docstring, 
        exp_id=exp_id
    )
    build_functions_and_methods(repo_args)
    

def print_stats(exp_id):
    repo_list = os.listdir(REPOS_DIR)
    results_file = str(EXTRACTION_DIR / f"{exp_id}_extracted.json")
    results = json.load(open(results_file))
    repo_list_w_function = list({x["file"]["file_module"]["repo"]["local_repo_path"] for x in results})
    print(f"Successfully extracted at least one functions from {len(repo_list_w_function)}/{len(repo_list)} repos")
    
    with open("results/error_repos.txt", 'w') as fout:
        for repo in repo_list:
            if repo not in repo_list_w_function:
                fout.write(repo+'\n')
                
    covered_repos = [repo.replace("___",'/') for repo in repo_list_w_function]
    stats_by_repo_name = check_repo_stats(covered_repos)
    json.dump(stats_by_repo_name, open(f"results/{exp_id}_repo_stats_func_filter.json", 'w'), indent=4)
    # stats_by_repo_name = json.load(open(f"results/{exp_id}_repo_stats_func_filter.json"))
    
    num_setup = len([k for k,v in stats_by_repo_name.items() if v["setup_file"]])
    print(f"Num repos: {len(stats_by_repo_name)}/{len(covered_repos)}")
    print(f"Percent with setup files: {num_setup}/{len(stats_by_repo_name)}={num_setup/len(stats_by_repo_name)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_names_file", type=str, default='results/repo_names_sampled.txt')
    parser.add_argument("--exp_id", type=str, default='temp')
    parser.add_argument('--disable_no_docstring', action='store_true')
    args = parser.parse_args()

    # load repo_list
    repo_list = open(args.repo_names_file).read().split('\n')
    repo_list = [x for x in repo_list if x]
    
    # clone repos
    print(f"Cloning {len(repo_list)} repos to {REPOS_DIR}..")
    for full_repo_name in tqdm(repo_list):
        clone_repo_from_url(full_repo_name)
    
    # get functions
    print(f"Getting functions..")
    extract_functions(disable_no_docstring=args.disable_no_docstring, exp_id=args.exp_id)
    
    print(f"Geting stats..")
    print_stats(args.exp_id)