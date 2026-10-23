"""Shared survey schema and fixed modeling definitions; no heavy imports."""
COLUMNS = ['location','age','sex','education','experience','primary_livelihood',
    'farm_size','variety','tree_age','spacing','intercrop','pruning','fertilizer',
    'pest_management','irrigation','shade','harvests_per_year','yield_original',
    'main_challenge','products_sold','buyer','soil_type','soil_fertility',
    'climate_issues','flood_or_drought','soil_management','household_size',
    'income_source','training_attendance','farm_support','expand_plan',
    'agency','agency_duration','agency_support','assistance_frequency','agency_influence']
MULTI = {'variety','intercrop','products_sold','buyer','income_source',
         'agency','agency_support'}
YIELD_ORDER_ORIGINAL = ['less than 100 kg','100-250 kg','251-500 kg','more than 750 kg']
YIELD_CLASSES = ['Less than 100 kg','100-250 kg','251 kg and above']
EXCLUDED = ['harvests_per_year','yield_original','products_sold','buyer',
            'income_source','expand_plan','agency_influence','assistance_frequency']
FEATURES = [c for c in COLUMNS if c not in EXCLUDED]
SEED = 42
CORE_MODELS = ["Cluster + decision tree", "Logistic regression", "Random forest", "Support vector machine"]
