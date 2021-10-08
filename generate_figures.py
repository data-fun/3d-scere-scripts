import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import os
import sqlite3
from statsmodels.stats.multitest import multipletests

###I- TRN download

def get_locus_info(database, query):
    """Query the SQLite database.

    Parameters
    ----------
    database : str
        Path to the SQLite database.
    query : str
        SQL query.

    Returns
    -------
    Pandas Dataframe
    """
    # Connect to database.
    db_connexion = sqlite3.connect(database)
    cursor = db_connexion.cursor()

    # Query database.
    chrom_info = cursor.execute(query)

    # Convert to Pandas dataframe
    column_names = [column[0] for column in chrom_info.description]
    chrom_info_df = pd.DataFrame(chrom_info.fetchall(), columns=column_names)

    # Select only strands + and -
    chrom_info_df = chrom_info_df[ (chrom_info_df["Strand"] == "C") | (chrom_info_df["Strand"] == "W") ]
    # Remove "2-micron" plasmid
    chrom_info_df = chrom_info_df[ chrom_info_df["Chromosome"] != "2-micron" ]
    # Convert chromosome id to int
    chrom_info_df["Chromosome"] = chrom_info_df["Chromosome"].astype(int)

    return chrom_info_df

#https://rdcu.be/cpoEH
#https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-74043-7/MediaObjects/41598_2020_74043_MOESM1_ESM.zip
Yeast_TRN = pd.read_csv("yeast2019-full-conds-net.csv", sep="\t", header = None)
Yeast_TRN.columns = ["TF", "TG", "type", "description"]
print(len(Yeast_TRN.TF.unique()))
#len(list_TF) = 220
print(len(Yeast_TRN.TG.unique()))
#len(Yeast_TRN.TG.unique()) = 6886
print(len(Yeast_TRN))
# len(Yeast_TRN) = 195498

Yeast_TRN = Yeast_TRN[Yeast_TRN.type.isin(["Direct", "Direct Positive", "Direct Dual", "Direct Negative"])]


sql_query = \
"""SELECT Standard_gene_name, Feature_name, Chromosome, Strand
FROM SGD_features
"""
bin_number = 50

# Get all features for all gene
loci = get_locus_info("SCERE.db", sql_query)


###II- Targets lists creation

list_TF = Yeast_TRN.TF.unique()
print(len(list_TF))
#len(list_TF) = 176
print(len(Yeast_TRN.TG.unique()))
#len(Yeast_TRN.TG.unique()) = 6175
print(len(Yeast_TRN))
# len(Yeast_TRN) = 45209

for TF in list_TF :

    TG = Yeast_TRN[Yeast_TRN.TF == TF]
    TG = TG.drop(["TF", "type", "description"], axis = 1)

    save = TG[TG["TG"].isin(loci["Feature_name"])]
    TG = TG.merge(loci, left_on = "TG", right_on = "Standard_gene_name")
    TG = TG["Feature_name"]
    TG = TG.append(save["TG"])
    TG = pd.DataFrame(TG)
    TG.columns = ["TG"]
    TG.to_csv("./TF_target_TRN_2019_AandB/" + str(TF) + "_" + str(len(TG)) + "_targets.csv",
              index = False)

#III- Cumulative distribution function and distribution histogram

def get_edges_list(gene_list, edges_list, feature_name):

    # Add SGDID
    feature_name = feature_name.merge(gene_list, left_on = "Feature_name", right_on = gene_list.columns[0])

    # Extract distances for selected genes list
    edges_list_select = edges_list[edges_list["Primary_SGDID"].isin(feature_name["Primary_SGDID"])]
    edges_list_select = edges_list_select[edges_list_select["Primary_SGDID_bis"].isin(feature_name["Primary_SGDID"])]
    edges_list_select.index = range(1, len(edges_list_select) + 1)

    return edges_list_select

def distri(genes_list, edges_list, feature_name, all_x, H2, F2, bin_number):

    edges_list_select = get_edges_list(genes_list, edges_list, feature_name)
    x = list(edges_list_select["3D_distances"])
    H, X1 = np.histogram(x, bins = bin_number, range = (0, 200))
    F1 = np.cumsum(H)/len(x)

    H = H/len(x)

    fig, ax = plt.subplots()
    ax.hist(X1[:-1], X1, weights=H2, color="#5767FF", alpha=0.3, label="All distances")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.09)
    plt.xlabel("3D distances", size = 16)
    ax.hist(X1[:-1], X1, weights=H, color="#FA3824", alpha=0.3, label="TF's targets")
    plt.ylabel("Density", size = 16)
    ax2=ax.twinx()
    plt.ylabel("CDF", size = 16)
    ax2.plot(X1[:-1], F1, label="CDF (TF's targets)", color = "#FA3824")
    ax2.plot(X1[:-1], F2, label="CDF (all)", color = "#5767FF")
    ax.legend(bbox_to_anchor = (0.6, 0.9), loc="upper left")
    ax2.legend(bbox_to_anchor = (0.6, 0.7), loc="upper left")

    # Compute Kolmogorov-Smirnov test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    if x != []:
    #if False:
        ks_result = ks_2samp(x, all_x)
    else:
        ks_result = None

    return fig, ks_result

sql_query = \
"""SELECT Primary_SGDID, Standard_gene_name, Chromosome, Feature_name, Strand, Stop_coordinate, Start_coordinate, Description
FROM SGD_features
"""

# Get all features for all gene
feature_name = get_locus_info("SCERE.db", sql_query)

edges_list = pd.read_parquet("3D_distances.parquet.gzip", engine="pyarrow")

#For all modules

files_names = os.listdir("./TF_target_TRN_2019_AandB")
results = pd.DataFrame(columns = ["TF", "description", "targets_number", "KS_stat", "KS_pvalue"])

all_x = list(edges_list["3D_distances"])
H2, X = np.histogram(all_x, bins = bin_number, range = (0, 200))
F2 = np.cumsum(H2)/sum(H2)
H2 = H2/len(all_x)

for file_name in files_names:
    print(file_name)
    FT_name = file_name.split("_")
    FT_name = FT_name[0]
    genes_list = pd.read_csv("./TF_target_TRN_2019_AandB/" + file_name, sep=",", header = [0])

    res = distri(genes_list, edges_list, feature_name, all_x, H2, F2, bin_number)
    if FT_name in feature_name["Standard_gene_name"].values:
        description = feature_name["Description"][feature_name["Standard_gene_name"] == FT_name].values[0]
    else:
        print(f"{FT_name} not in data !")
        description = None

    if res[1] != None:
        results = results.append({"TF": FT_name,
                                  "Description": description,
                                  "Targets_number": len(genes_list),
                                  "KS_stat": res[1].statistic,
                                  "KS_pvalue": res[1].pvalue},
                                 ignore_index=True)

    else:
        results = results.append({"TF": FT_name,
                                  "Description": description,
                                  "Targets_number": len(genes_list),
                                  "KS_stat": None,
                                  "KS_pvalue": 1},
                                 ignore_index=True)

    res[0].savefig(f"./distri_2019_EandB_FV/{file_name}.png")
    plt.close()

multitests_correction = multipletests(results["KS_pvalue"], method='bonferroni')[1]
results["KS_pvalue_adj"] = pd.Series(multitests_correction)

results.to_csv("3D_distances_results_recapitulation_2019_EandB_BF_cor.csv",
               sep = "\t",
               index = False,
               columns = ["TF", "Description", "Targets_number", "KS_stat", "KS_pvalue", "KS_pvalue_adj"])