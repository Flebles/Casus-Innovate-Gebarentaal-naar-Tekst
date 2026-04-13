#!/usr/bin/env python3

import argparse
from src.gesture import GestureModelTrainer


# Train gesture recognition model
def train_model(dataset_path: str = "data/gestures.csv", model_path: str = "models/gesture_model.pkl"):

    print("\n" + "="*60)
    print("Training Gesture Recognition Model")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Model output: {model_path}")
    print("="*60 + "\n")

    trainer = GestureModelTrainer(dataset_path)

    try:
        # Train the model
        metrics = trainer.train()
        trainer.save_model(model_path)

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall:    {metrics['recall']:.2%}")
        print(f"F1-Score:  {metrics['f1_score']:.2%}")
        print(f"\nGestures learned: {', '.join(sorted(metrics['classes']))}")
        print(f"Model saved to: {model_path}")

    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Please collect data first with main_collect.py")
        return False
    
    except Exception as e:
        print(f"ERROR: Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train gesture recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_train.py --dataset data/gestures.csv --model models/ngt_gesture_model.pkl
  python main_train.py --dataset data/numbers.csv --model models/numbers_model.pkl
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/gestures.csv",
        help="Dataset CSV path (default: data/gestures.csv)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/gesture_model.pkl",
        help="Model output path (default: models/gesture_model.pkl)"
    )

    args = parser.parse_args()

    success = train_model(args.dataset, args.model)

    if not success:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

