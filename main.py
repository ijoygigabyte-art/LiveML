from data_loader import load_and_clean_data
import problem1
import problem2
import problem3
import problem4
import problem5

def main():
    print("======================================================")
    print("ASDS 6302 - Assignment 2 Unified Pipeline")
    print("Dataset: UCI Bike Sharing")
    print("======================================================\n")
    
    # 1. Load Data
    print("Initializing Data Loader...")
    df = load_and_clean_data()
    if df is None:
        return
    print(f"Data Successfully Loaded. Shape: {df.shape}\n")

    # 2. Run Pipeline Modules
    problem1.run(df)
    problem2.run(df)
    problem3.run(df)
    problem4.run(df)
    problem5.run(df)

    print("======================================================")
    print("Assignment 2 Pipeline Execution Complete!")
    print("All models, metrics, and plots have been generated.")
    print("======================================================")

if __name__ == "__main__":
    main()
