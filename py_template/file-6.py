from data_loader import collect_all_data
from classifier import train_classifier_cv

def main():
    print("## Human Activity Learning Using Mobile Phone Data ##")
    
    # 1. Load Data
    tx, ty, tz, labels = collect_all_data()
    
    if tx is None:
        return

    # 2. Train and Validate
    _, accuracy = train_classifier_cv(tx, ty, tz, labels)
    
    print(f"\nFinal Validation Accuracy: {accuracy:.4f} ({(accuracy*100):.2f}%)")

if __name__ == "__main__":
    main()
