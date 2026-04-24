import json
import re
from my_project.crew import MyProjectCrew
from crewai.flow.flow import Flow, start, listen

class YelpPredictionFlow(Flow):
    @start()
    def load_test_data(self):
        with open("data/test_review_subset.json", 'r') as f:
            return json.loads(f.readline())

    @listen(load_test_data)
    def run_prediction_crew(self, test_case):
        inputs = {'user_id': test_case['user_id'], 'item_id': test_case['item_id']}
        MyProjectCrew().hierarchical_crew().kickoff()

    @listen(run_prediction_crew)
    def save_report(self, result):
        #logic save to report.json
        print("Flow Complete!")

#this is to run: YelpPredictionFlow().kickoff()
def extract_json_from_output(raw_output: str) -> dict:
    """Sanitize LLM output to ensure we get valid JSON."""
    text = str(raw_output).strip()
    text = text.replace('{{', '{').replace('}}', '}')
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"stars": 0.0, "review": text, "_error": "Failed to parse JSON"}

def run():
    #loading the test case
    test_json_path = "data/test_review_subset.json"
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_case = json.loads(f.readline())

    inputs = {
        'user_id': test_case['user_id'],
        'item_id': test_case['item_id']
    }

    print(f"--- Running Prediction for User: {inputs['user_id']} ---")

    #kickoff the Crew
    try:
        result = MyProjectCrew().sequential_crew().kickoff(inputs=inputs)
        
        #saving the result
        report = extract_json_from_output(result.raw)
        with open('report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDONE! Prediction: {report.get('stars')} stars.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
