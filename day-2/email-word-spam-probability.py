def get_validated_input(prompt, min_val=0, max_val=None):
    """
    Prompts the user for an integer input and validates it.

    Args:
        prompt (str): The message to display to the user.
        min_val (int): The minimum allowed value for the input.
        max_val (int): The maximum allowed value for the input (inclusive).

    Returns:
        int: The validated integer input.
    """
    while True:
        try:
            value = int(input(prompt))
            if value < min_val:
                print(f"Error: Please enter a value greater than or equal to {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Error: Please enter a value less than or equal to {max_val}.")
            else:
                return value
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")

def calculate_spam_probability():
    """
    Calculates the probability of an email being spam given it contains "free"
    using Bayes' Theorem based on user-provided data.
    """
    print("Please enter the following data for your email dataset:")

    total_emails = get_validated_input("Total number of emails: ")
    emails_with_free = get_validated_input("Number of emails containing the word 'free': ", max_val=total_emails)
    spam_emails = get_validated_input("Number of spam emails: ", max_val=total_emails)
    spam_and_free = get_validated_input("Number of emails that are both spam and contain 'free': ",
                                        max_val=min(emails_with_free, spam_emails))

    print("\n--- Calculating Probabilities ---")

    # P(Spam) = spam_emails / total_emails
    if total_emails == 0:
        print("Error: Total number of emails cannot be zero.")
        return

    p_spam = spam_emails / total_emails
    print(f"P(Spam) = {p_spam:.4f}")

    # P(Free) = emails_with_free / total_emails
    if emails_with_free == 0:
        print("Error: P(Free) is zero. Cannot compute P(Spam | Free) as there are no emails with 'free'.")
        return

    p_free = emails_with_free / total_emails
    print(f"P(Free) = {p_free:.4f}")

    # P(Free | Spam) = spam_and_free / spam_emails
    if spam_emails == 0:
        # If there are no spam emails, then P(Free | Spam) is undefined or 0 (depending on interpretation).
        # In this context, if spam_emails is 0, it implies P(Spam) is 0, and Bayes' Theorem isn't directly applicable for a non-zero P(Free).
        # We'll set it to 0 as no spam emails means no free in spam.
        p_free_given_spam = 0.0
        print("Warning: No spam emails found. P(Free | Spam) is considered 0.")
    else:
        p_free_given_spam = spam_and_free / spam_emails
        print(f"P(Free | Spam) = {p_free_given_spam:.4f}")

    # Bayes' Theorem: P(Spam | Free) = P(Free | Spam) * P(Spam) / P(Free)
    if p_free == 0:
        # This case is already handled above, but as a double-check for the formula.
        print("Error: Division by zero (P(Free) is zero). Cannot compute P(Spam | Free).")
        return

    p_spam_given_free = (p_free_given_spam * p_spam) / p_free

    print(f"\nP(Spam | Free): {p_spam_given_free:.4f}")

if __name__ == "__main__":
    calculate_spam_probability()