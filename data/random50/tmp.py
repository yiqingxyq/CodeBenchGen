import json 

data = json.load(open("cleaned_python_test.json"))

for round_id in range(3):
    for suffix in [".json", "_exec.json"]:
        filename = f"cleaned_python_test_main_round{round_id}{suffix}"
        new_data = json.load(open(filename))
        
        for example, new_example in zip(data, new_data):
            for k in ["repo_name", "func_path"]:
                new_example[k] = example[k]
                
        json.dump(new_data, open(filename, "w"), indent=4)