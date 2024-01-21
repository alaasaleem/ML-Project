import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataFrame = pd.read_csv("smoking.csv")  # load csv file

categorical_columns = ['gender', 'oral', 'dental caries', 'tartar']  # get some categorical values

numeric_columns = ['weight(kg)', 'age', 'Cholesterol', 'triglyceride', 'fasting blood sugar', 'hemoglobin']  # add 'hemoglobin'
numeric_data = dataFrame[numeric_columns]

colors = sns.color_palette("Set2")

plt.rcParams['font.family'] = 'Times New Roman'  # make the font in Times New Roman

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), facecolor='black', gridspec_kw={'hspace': 0.5})  # add gridspec_kw

# Plot the bar charts
for i, column in enumerate(categorical_columns):
    row, col = divmod(i, 2)
    counts = dataFrame[column].value_counts()
    bars = counts.plot(kind='bar', ax=axes[row, col], color=colors, legend=False, edgecolor='white')
    axes[row, col].set_title(f'Bar Chart of  "{column.capitalize()}"  Feature', color='white')  # Make the title in white

    # Add frequencies inside the rectangles + white border
    for bar, freq in zip(bars.patches, counts):
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width() / 2, height / 2, f'{freq}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white',
                            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'))

    # Rotate the x-axis labels and make the color in white
    axes[row, col].tick_params(axis='x', rotation=0, colors='white')
    axes[row, col].set_xlabel(column.capitalize(), color='white')  # Set x-axis label color to 'white'

    axes[row, col].set_ylabel('Count', color='white')
    axes[row, col].tick_params(axis='y', colors='white')

    axes[row, col].spines['bottom'].set_color('white')
    axes[row, col].spines['top'].set_color('white')
    axes[row, col].spines['right'].set_color('white')
    axes[row, col].spines['left'].set_color('white')

    axes[row, col].tick_params(axis='x', colors='white')
    axes[row, col].tick_params(axis='y', colors='white')

    # Make the background color in black for each subplot
    axes[row, col].set_facecolor('black')

    # Calculate the mode
    mode_value = counts.idxmax()

    # Add mode text in a box in the upper right corner
    mode_text = f'Mode: {mode_value}'
    axes[row, col].text(0.95, 0.95, mode_text, transform=axes[row, col].transAxes, color='white',
                        fontsize=10, ha='right', va='top',
                        bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3'))

# Make the background color for the figure
fig.patch.set_facecolor('black')

# Set up the style
sns.set(style="whitegrid")

# Create subplots for histograms
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), facecolor='black', gridspec_kw={'hspace': 0.5})  # add gridspec_kw

# Plot histograms
for i, column in enumerate(numeric_columns):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    ax.hist(numeric_data[column], bins=20, color='skyblue', edgecolor='white')

    # Calculate the mean for each column
    mean_value = numeric_data[column].mean()

    # Add correct mean text in a box and a red line in the upper right corner
    mean_text = f'Mean: {mean_value:.2f}'
    ax.text(0.95, 0.95, mean_text, transform=ax.transAxes, color='white', fontsize=10, ha='right', va='top',
            bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3'))

    # Add a red line representing the mean
    ax.axvline(x=mean_value, color='red', linestyle='--')

    ax.set_title(f'Histogram of {column.capitalize()}', color='white')
    ax.set_xlabel(column.capitalize(), color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.tick_params(axis='both', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_facecolor('black')

# Adjust layout
plt.tight_layout()

# Increase space between y-axis labels and nearby plots
plt.subplots_adjust(wspace=0.5)

# Show the plot
plt.show()
