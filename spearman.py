# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr

# artifact_intensities = np.array([0.005, 0.001, 0.05, 0.1])

# # from your bar chart (approx values)
# measured_values = np.array([28.9, 28.0, 23.2, 21.0])

# # from JSON results
# measured_artifacts = np.array([0.961879,0.972471, 0.889025, 0.763031])

# # put into DataFrame
# df = pd.DataFrame({
#     "measured_values": measured_values,
#     "measured_artifacts": measured_artifacts
# })

# # compute Spearman correlation
# corr, pval = spearmanr(df["measured_values"], df["measured_artifacts"])
# print(f"Spearman correlation: {corr:.4f}, p-value: {pval:.4g}")

# # correlation matrix for heatmap
# corr_matrix = df.corr(method="spearman")

# # plot heatmap
# plt.figure(figsize=(6,5))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
# plt.title("Spearman Correlation Heatmap")
# plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

artifact_intensities = np.array([0.005, 0.001, 0.05, 0.1])
artifact_intensities = 1 - artifact_intensities  # Invert to match bar chart


# from bar chart
measured_values = np.array([28.9, 28.0, 23.2, 21.0])

# from JSON results
measured_artifacts = np.array([0.961879, 0.972471, 0.889025, 0.763031])

# put into DataFrame
df = pd.DataFrame({
    "Artifact Degradation": artifact_intensities,
    "Success Rate %": measured_values,
    "Artifact Score (E3D)": measured_artifacts
})

# compute Spearman correlation matrix
corr_matrix = df.corr(method="spearman")
print(corr_matrix)

# plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Spearman Correlation Artifacts")
plt.show()