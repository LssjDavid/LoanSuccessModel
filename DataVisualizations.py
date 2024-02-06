for column in bank.select_dtypes(include='object').columns:
    print(column)
    print(bank[column].unique())

categorical_cols = []
for feature in bank.columns:
    if ((bank[feature].dtypes == 'O') and (feature not in ['deposits'])):
        categorical_cols.append(feature)

#Just checking all the categorial features so we can create visualizations
print(categorical_cols)

print(bank.columns)

num_plots = len(categorical_cols) #Number of loops to run through
num_cols = 3  # Number of columns in each row
num_rows = int(np.ceil(num_plots / num_cols))  #Calculates the number of rows

#Increases the width to avoid overcramping
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 120), facecolor='white')

#flatten the axes if there is more than one row
axes = axes.flatten()

#Loops through the categorical features
for i, categorical_feature in enumerate(categorical_cols):
    sns.countplot(x=categorical_feature, data=bank, ax=axes[i], orient='v')
    axes[i].set_ylabel('Count')
    axes[i].set_xlabel(categorical_feature)
    axes[i].set_title(categorical_feature)
    axes[i].tick_params(axis='x', rotation=45)  #Rotate x-axis labels since the job names overlapped

#Hide any empty subplots
for j in range(num_plots, num_rows * num_cols):
    axes[j].axis('off')

#Show the plots
plt.tight_layout()
plt.show()
