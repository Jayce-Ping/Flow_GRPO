import json
import re


def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))


    
prompt_file_path = 'dataset/T2IS/prompt.json'



data = json.load(open(prompt_file_path))


# Group data by its grid form
groups = {}
for item in data:
    grid_info = extract_grid_info(item['prompt'])
    if grid_info not in groups:
        groups[grid_info] = []
    
    groups[grid_info].append(item)


square_2x2 = groups[(2,2)]

with open('dataset/T2IS/train_metadata.jsonl', 'w') as f:
    for item in square_2x2:
        f.write(json.dumps(item) + '\n')


with open('dataset/T2IS/test_metadata.jsonl', 'w') as f:
    for item in square_2x2:
        f.write(json.dumps(item) + '\n')