dataset_map = {
    "lucas_full":"LUCAS",
    "lucas_skipped":"LUCAS (Skipped)",
    "lucas_downsampled":"LUCAS (Downsampled)",
    "lucas_min":"LUCAS (Truncated)",
    "indian_pines":"Indian Pines",
    "ghsi":"GHSI",
}

metric_map = {
    "time":{
        "LUCAS": "Time (seconds)",
        "LUCAS (Skipped)": "Time (seconds)",
        "LUCAS (Downsampled)": "Time (seconds)",
        "LUCAS (Truncated)": "Time (seconds)",
        "Indian Pines": "Time (seconds)",
        "GHSI": "Time (seconds)",
    },
    "metric1":{
        "LUCAS": "$R^2$",
        "LUCAS (Skipped)": "$R^2$",
        "LUCAS (Downsampled)": "$R^2$",
        "LUCAS (Truncated)": "$R^2$",
        "Indian Pines": "OA (%)",
        "GHSI": "OA (%)",
    },
    "metric2":{
        "LUCAS": "$RMSE$",
        "LUCAS (Skipped)": "$RMSE$",
        "LUCAS (Downsampled)": "$RMSE$",
        "LUCAS (Truncated)": "$RMSE$",
        "Indian Pines": "$\kappa$",
        "GHSI": "$\kappa$",
    }
}

algorithm_map = {
    "all_bands" : "All Bands",
    "fsdrl":"BSDR",
    "bsnet":"BS-Net-FC",
    "zhang":"Zhang et al.",
    "mcuve":"MCUVE",
    "pcal":"PCA-loading",
    "lasso":"LASSO",
    "spa":"SPA",
}