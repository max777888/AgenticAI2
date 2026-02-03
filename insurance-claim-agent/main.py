from workflow import ClaimProcessingWorkflow
from claim_storage import ClaimStorage


def main():
    print("Insurance Claim Assistant  (class-based version)")
    print("Commands:  submit   ask   quit / q / exit\n")

    storage = ClaimStorage()
    workflow = ClaimProcessingWorkflow()

    while True:
        cmd = input("> ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        elif cmd == "submit":
            text = input("Claim description:\n> ").strip()
            if text:
                claim_id = storage.submit_claim(text)
                print(f"✓ Claim registered  |  ID: {claim_id}")
            else:
                print("Description cannot be empty.")

        elif cmd == "ask":
            question = input("Your question:\n> ").strip()
            if question:
                print("\nThinking...\n")
                answer = workflow.run(question)
                print("Answer:")
                print(answer)
                print("─" * 70)
            else:
                print("Please ask something.")

        else:
            print("Available commands:  submit  ask  quit")


if __name__ == "__main__":
    main()