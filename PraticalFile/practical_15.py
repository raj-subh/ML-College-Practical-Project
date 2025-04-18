import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactional data
data = [
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Eggs"],
]

# Convert to one-hot encoded dataframe
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit_transform(data)
df = pd.DataFrame(te_array, columns=te.columns_)

# Step 1: Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display
print(frequent_itemsets)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

"""
🔍 Output Explanation:
Antecedents: Items on the left-hand side of the rule.

Consequents: Items on the right-hand side of the rule.

Support: How frequently the itemset appears in the dataset.

Confidence: Likelihood that the rule is correct.

Lift: Strength of the rule over random chance (>1 means strong association). 
"""
