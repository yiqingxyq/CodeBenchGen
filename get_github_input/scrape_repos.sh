# curl -H "Authorization: token ${GITHUB_TOKEN}" \
#     "https://api.github.com/search/repositories?q=language:Python+license:mit+created:>2024-10-01&sort=updated&order=desc&per_page=100&page=$page" \
#     | jq -r '.items[] | .full_name' > repo_full_names.txt;

for page in {1..10}; do
  curl -H "Authorization: token ${GITHUB_TOKEN}" \
       "https://api.github.com/search/repositories?q=language:Python+license:mit+created:>2024-10-01&sort=updated&order=desc&per_page=100&page=$page" \
       | jq -r '.items[] | .full_name' >> repo_names.txt;
done

python sample_repo.py --sampled_size 50