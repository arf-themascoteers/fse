import pandas as pd


def remove_mcuve():
    summary = 'saved\\1_all_ex_spa\\fsdr.csv'
    details = 'saved\\1_all_ex_spa\\details_fsdr.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas_full")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas_full")]
    details_df.to_csv(details, index=False)


def remove_mcuve2():
    summary = 'saved\\1_mcuve_spa\\mcuve_luc.csv'
    details = 'saved\\1_mcuve_spa\\details_mcuve_luc.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)


def remove_lucasmin():
    summary = 'saved\\1_all_ex_spa\\fsdr.csv'
    details = 'saved\\1_all_ex_spa\\details_fsdr.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[ (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)

def remove_mcuve_lucasmin():
    summary = 'saved\\1_spa_luc\\mcuve_luc.csv'
    details = 'saved\\1_spa_luc\\details_mcuve_luc.csv'

    summary_df = pd.read_csv(summary)
    details_df = pd.read_csv(details)

    summary_df = summary_df[ (summary_df["algorithm"] != "mcuve") | (summary_df["dataset"] != "lucas_min")]
    summary_df.to_csv(summary, index=False)

    details_df = details_df[(details_df["algorithm"] != "mcuve") | (details_df["dataset"] != "lucas_min")]
    details_df.to_csv(details, index=False)

remove_lucasmin()