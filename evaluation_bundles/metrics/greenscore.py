from green_score import GREEN
import os
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
import json

class GreenScorer:
    def __init__(self, model_name : str = "StanfordAIMI/GREEN-radllama2-7b"):
        self.model = model_name
        self.green_scorer = GREEN(self.model, output_dir=".")

    def calculate_metric(self, references: list, hypotheses: list) -> tuple:
        mean, std, green_score_list, summary, result_df = self.green_scorer(references, hypotheses)
        return mean, std, green_score_list, summary, result_df


# test
if __name__=="__main__":

    USE_CASE_ROOT = "/import/nlp/multimodal_report_generation/"

    dataset = "xraygpt_reports_without_indication_ReXGradient-160K_30samples_subset_only_lateral_and_frontal_views"

    with open(os.path.join(USE_CASE_ROOT,f"reports/{dataset}.json"),"r") as f:
        generated_reports = json.load(f)
    
    with open(os.path.join(USE_CASE_ROOT,"ReXGradient-160K_30samples_subset_only_lateral_and_frontal_views_gold_reports.json"),"r") as f:
        gold_reports = json.load(f)
    
    # for testing extract three keys from generated_reports
    test_keys = list(generated_reports.keys())

    test_generated = [generated_reports[key] for key in test_keys]
    test_gold = [gold_reports[key] for key in test_keys]

    # initiate scorer
    scorer = GreenScorer()

    mean, std, green_score_list, summary, result_df = scorer.calculate_metric(test_gold, test_generated)

    result_df.index = test_keys

    results_obj = result_df.to_dict(orient="index")

    # add mean and std and summary
    export_obj = {}
    export_obj["green_score"] = {"mean": mean, "std": std}
    export_obj["summary"] = summary
    export_obj["details"] = results_obj

    # Save results as json
    with open(os.path.join(USE_CASE_ROOT,f"evaluation/green_score_{dataset}.json"),"w") as f:
        json.dump(export_obj, f, indent=4)

    print("Saved results!")

    # print(green_score_list)
    # print(summary)
    # for index, row in result_df.iterrows():
    #     print(f"Row {index}:\n")
    #     for col_name in result_df.columns:
    #         print(f"{col_name}: {row[col_name]}\n")
    #     print('-' * 80)


