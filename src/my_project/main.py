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

def run(crew_type="Sequential"):#defualt is the sequential
    test_json_path = "data/test_review_subset.json"
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_case = json.loads(f.readline())

    inputs = {'user_id': test_case['user_id'], 'item_id': test_case['item_id']}
    
    #initialize the project
    project = MyProjectCrew()

    try:
        #determining which crew to kickoff based on the pattern you want to test
        if crew_type == "Hierarchical":
            result = project.hierarchical_crew().kickoff(inputs=inputs)
        else:
            result = project.sequential_crew().kickoff(inputs=inputs)

        prediction_data = extract_json_from_output(result.raw)
        task_results = result.tasks_output

        formatted_report = {
            "execution_metadata": {
                "process_pattern": crew_type,
                "model_used": "ollama/phi3",
                "user_id": inputs['user_id']
            },
            "prediction_output": prediction_data,
            "pattern_breakdown": {
                # Single Task Pattern
                "final_synthesis_pattern": "Collaborative Single Task", 
                "individual_contributions": {
                    "user_profiler_output": task_results[0].raw if len(task_results) > 0 else "N/A",
                    "item_analyst_output": task_results[1].raw if len(task_results) > 1 else "N/A",
                    "eda_specialist_output": task_results[2].raw if len(task_results) > 2 else "N/A"
                }
            }
        }

        with open('report.json', 'w', encoding='utf-8') as f:
            json.dump(formatted_report, f, indent=2, ensure_ascii=False)
            
        print(f"Report saved! Pattern used: {crew_type}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run(crew_type="Hierarchical")