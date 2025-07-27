import random

def simulate_dice_rolls(num_trials=10000):
    """
    Simulates rolling two 6-sided dice a specified number of times
    and estimates probabilities for specific sums.

    Args:
        num_trials (int): The number of times to simulate rolling the dice.
    """
    count_sum_7 = 0
    count_sum_2 = 0
    count_sum_gt_10 = 0

    for _ in range(num_trials):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        current_sum = die1 + die2

        if current_sum == 7:
            count_sum_7 += 1
        elif current_sum == 2:
            count_sum_2 += 1
        elif current_sum > 10:  # This covers sums of 11 and 12
            count_sum_gt_10 += 1

    # Calculate estimated probabilities
    prob_sum_7 = count_sum_7 / num_trials
    prob_sum_2 = count_sum_2 / num_trials
    prob_sum_gt_10 = count_sum_gt_10 / num_trials

    # Print the results
    print(f"P(Sum = 7): {prob_sum_7:.4f}")
    print(f"P(Sum = 2): {prob_sum_2:.4f}")
    print(f"P(Sum > 10): {prob_sum_gt_10:.4f}")

if __name__ == "__main__":
    simulate_dice_rolls()