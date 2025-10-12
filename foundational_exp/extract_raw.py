import json

with open('./AgentJudge-strict.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

extracted_data = []
for item in data:
    contents = item['contents']
    raw_record = ""
    for content in contents:
        for turn in content:
            if turn['role'] == 'user':
                query = turn['content']
                raw_record += f"Query: {query}\n"
            elif turn['role'] == 'agent':
                thought = turn['thought']
                action = turn['action']
                raw_record += f"Thought: {thought}\nAction: {action}\n"
    
    extracted_data.append({
        "id": item['id'],
        "profile": item['profile'],
        "raw_record": raw_record.strip(),
        "label": item['label'],
        "application_scenario": item['application_scenario'],
        "risk_type": item['risk_type'],
        "failure_mode": item['failure_mode'],
        "ambiguous": item['ambiguous'],
        "risk_description": item['risk_description'],
    })

with open('./AgentJudge-strict-raw.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)