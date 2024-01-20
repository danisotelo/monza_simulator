import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores, another_vector):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Create a subplot for the scores and mean_scores
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.title('Scores and Mean Scores')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    #plt.ylim(ymin=0)
    plt.legend()

    # Add text labels for scores and mean_scores
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # Create a subplot for the new vector
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.title('Reward')
    plt.xlabel('dt')
    plt.ylabel('Value')
    plt.plot(another_vector, label='Reward', color='green')
    #plt.ylim(ymin=0)
    plt.legend()

    # Add a text label for the new vector
    plt.text(len(another_vector)-1, another_vector[-1], str(another_vector[-1]))

    plt.tight_layout()  # Adjust layout
    plt.show(block=False)
    plt.pause(.1)