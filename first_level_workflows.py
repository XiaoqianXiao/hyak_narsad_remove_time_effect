#Nipype v1.10.0.
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _dict_ds_lss
from utils import _bids2nipypeinfo
from utils import _bids2nipypeinfo_lss
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design, FEATModel
import logging

# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class DerivativesDataSink(BIDSDerivatives):
    """Custom data sink for first-level analysis outputs."""
    out_path_base = 'firstLevel_timeEffect'

DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']
DATA_ITEMS_LSS = ['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']

# Available contrast patterns
CONTRAST_PATTERNS = {
    'all_vs_baseline': 'Each condition vs baseline',
    'pairwise': 'All pairwise comparisons between conditions',
    'first_vs_rest': 'First condition vs all others',
    'group_vs_group': 'First half vs second half of conditions',
    'linear_trend': 'Linear trend across conditions',
    'quadratic_trend': 'Quadratic trend across conditions',
    'specific_face_vs_house': 'Face vs house conditions (if present)',
    'direct_weights': 'Direct weight specification as list/tuple (e.g., [1, -1, 0])'
}

# Available contrast types
CONTRAST_TYPES = {
    'standard': 'Pairwise comparisons between all conditions',
    'minimal': 'Each condition vs baseline only',
    'custom': 'Use custom patterns defined in contrast_patterns'
}

# =============================================================================
# CONTRAST GENERATION FUNCTIONS
# =============================================================================

def extract_cs_conditions(df_trial_info):
    """
    Extract and group CS-, CSS, and CSR conditions from a pandas DataFrame.
    
    This function adds a 'conditions' column to the DataFrame that groups trials:
    - First trial of each CS type becomes 'CS-_first', 'CSS_first', 'CSR_first'
    - Remaining trials of each type become 'CS-_others', 'CSS_others', 'CSR_others'
    - All other trials keep their original trial_type as conditions value
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                        The 'trial_type' column contains condition names,
                                        and 'onset' column is used for chronological sorting.
    
    Returns:
        tuple: (df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions)
            - df_with_conditions: DataFrame with added 'conditions' column
            - cs_conditions: dict with 'first' and 'other' keys for CS- conditions
            - css_conditions: dict with 'first' and 'other' keys for CSS conditions  
            - csr_conditions: dict with 'first' and 'other' keys for CSR conditions
            - other_conditions: List of non-CS/CSS/CSR conditions
    """
    import pandas as pd
    
    # Validate DataFrame input
    if not isinstance(df_trial_info, pd.DataFrame):
        raise ValueError("df_trial_info must be a pandas DataFrame")
    
    if df_trial_info.empty:
        raise ValueError("DataFrame cannot be empty")
    
    required_columns = ['trial_type', 'onset']
    missing_columns = [col for col in required_columns if col not in df_trial_info.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying original
    df_work = df_trial_info.copy()
    
    # Initialize conditions column with trial_type values
    df_work['conditions'] = df_work['trial_type'].copy()
    
    logger.info(f"Using DataFrame input with {len(df_work)} trials")
    logger.info(f"DataFrame columns: {list(df_work.columns)}")
    
    # Find first trial of each CS type (by onset time)
    cs_first_half_trials = df_work[df_work['trial_type'].str.startswith('CS-_first_half') & 
                       ~df_work['trial_type'].str.startswith('CSS_first_half') & 
                       ~df_work['trial_type'].str.startswith('CSR_first_half')].copy()
    css_first_half_trials = df_work[df_work['trial_type'].str.startswith('CSS_first_half')].copy()
    csr_first_half_trials = df_work[df_work['trial_type'].str.startswith('CSR_first_half')].copy()
    
    # Update conditions column for CS- trials
    if not cs_first_half_trials.empty:
        cs_first_half_first_idx = cs_first_half_trials.sort_values('onset').index[0]
        df_work.loc[cs_first_half_first_idx, 'conditions'] = 'CS-_first_half_first'
        cs_first_half_other_indices = cs_first_half_trials[cs_first_half_trials.index != cs_first_half_first_idx].index
        df_work.loc[cs_first_half_other_indices, 'conditions'] = 'CS-_first_half_others'
        logger.info(f"CS-_first_half conditions: first trial at index {cs_first_half_first_idx}, {len(cs_first_half_other_indices)} others")
    
    # Update conditions column for CSS trials
    if not css_first_half_trials.empty:
        css_first_half_first_idx = css_first_half_trials.sort_values('onset').index[0]
        df_work.loc[css_first_half_first_idx, 'conditions'] = 'CSS_first_half_first'
        css_first_half_other_indices = css_first_half_trials[css_first_half_trials.index != css_first_half_first_idx].index
        df_work.loc[css_first_half_other_indices, 'conditions'] = 'CSS_first_half_others'
        logger.info(f"CSS_first_half conditions: first trial at index {css_first_half_first_idx}, {len(css_first_half_other_indices)} others")
    
    # Update conditions column for CSR trials
    if not csr_first_half_trials.empty:
        csr_first_half_first_idx = csr_first_half_trials.sort_values('onset').index[0]
        df_work.loc[csr_first_half_first_idx, 'conditions'] = 'CSR_first_half_first'
        csr_first_half_other_indices = csr_first_half_trials[csr_first_half_trials.index != csr_first_half_first_idx].index
        df_work.loc[csr_first_half_other_indices, 'conditions'] = 'CSR_first_half_others'
        logger.info(f"CSR_first_half conditions: first trial at index {csr_first_half_first_idx}, {len(csr_first_half_other_indices)} others")
    
    # Get unique conditions for contrast generation
    unique_conditions = df_work['conditions'].unique().tolist()
    logger.info(f"Unique conditions for contrast generation: {unique_conditions}")
    
    # Extract grouped conditions for backward compatibility
    cs_first_half_conditions = {'first': 'CS-_first_half_first' if 'CS-_first_half_first' in unique_conditions else None, 
                     'other': ['CS-_first_half_others'] if 'CS-_first_half_others' in unique_conditions else []}
    css_first_half_conditions = {'first': 'CSS_first_half_first' if 'CSS_first_half_first' in unique_conditions else None, 
                      'other': ['CSS_first_half_others'] if 'CSS_first_half_others' in unique_conditions else []}
    csr_first_half_conditions = {'first': 'CSR_first_half_first' if 'CSR_first_half_first' in unique_conditions else None, 
                      'other': ['CSR_first_half_others'] if 'CSR_first_half_others' in unique_conditions else []}
    
    # Get other conditions (non-CS/CSS/CSR)
    other_conditions = df_work[~df_work['trial_type'].str.startswith('CS')]['trial_type'].unique().tolist()
    
    logger.info(f"Processed conditions: CS-={cs_first_half_conditions}, CSS={css_first_half_conditions}, CSR={csr_first_half_conditions}")
    logger.info(f"Other conditions: {other_conditions}")
    
    return df_work, cs_first_half_conditions, css_first_half_conditions, csr_first_half_conditions, other_conditions


def create_contrasts(df_trial_info, contrast_type='interesting'):
    """
    Create contrasts dynamically based on DataFrame with CS-, CSS, and CSR grouping.
    
    This function identifies the first trial of each condition type (CS-, CSS, CSR) as 
    separate conditions, groups all other trials of each type together, and creates 
    contrasts for all other trial types as individual conditions.
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                        The 'trial_type' column contains condition names,
                                        and 'onset' column is used for chronological sorting.
        contrast_type (str): Type of contrasts to create ('interesting', 'standard', 'minimal', 'custom')
    
    Returns:
        tuple: (contrasts_list, cs_conditions, css_conditions, csr_conditions, other_conditions)
            - contrasts_list: List of contrast tuples (name, type, conditions, weights)
            - cs_conditions: dict with 'first' and 'other' keys for CS- conditions
            - css_conditions: dict with 'first' and 'other' keys for CSS conditions
            - csr_conditions: dict with 'first' and 'other' keys for CSR conditions
            - other_conditions: List of non-CS/CSS/CSR conditions
    """
    if df_trial_info is None:
        raise ValueError("df_trial_info is required")
    
    # Extract CS-, CSS, and CSR conditions with grouping
    df_with_conditions, cs_first_half_conditions, css_first_half_conditions, csr_first_half_conditions, other_conditions = extract_cs_conditions(df_trial_info)
    
    # Use the conditions column for contrast generation
    all_contrast_conditions = df_with_conditions['conditions'].unique().tolist()
    
    contrasts = []
    
    if contrast_type == 'interesting':
        # Use the focused interesting contrasts
        contrasts, cs_first_half_conditions, css_first_half_conditions, csr_first_half_conditions, other_conditions = create_interesting_contrasts(df_trial_info)
        return contrasts, cs_first_half_conditions, css_first_half_conditions, csr_first_half_conditions, other_conditions
    
    elif contrast_type == 'minimal':
        # Create simple contrasts for each condition vs baseline
        for condition in all_contrast_conditions:
            contrasts.append((f'{condition}>baseline', 'T', [condition], [1]))
    
    elif contrast_type == 'standard':
        # Create pairwise contrasts between all conditions
        for i, cond1 in enumerate(all_contrast_conditions):
            for j, cond2 in enumerate(all_contrast_conditions):
                if i < j:  # Avoid duplicate contrasts
                    contrasts.append((f'{cond1}>{cond2}', 'T', [cond1, cond2], [1, -1]))
                    contrasts.append((f'{cond1}<{cond2}', 'T', [cond1, cond2], [-1, 1]))
    
    elif contrast_type == 'custom':
        # Define custom contrasts based on condition patterns
        if 'first_half' in str(other_conditions) and 'second_half' in str(other_conditions):
            # Split conditions by halves
            first_half_conds = [c for c in other_conditions if 'first_half' in c]
            second_half_conds = [c for c in other_conditions if 'second_half' in c]
            
            # Create contrasts between halves
            for f_cond in first_half_conds:
                base_cond = f_cond.replace('first_half', 'second_half')
                if base_cond in second_half_conds:
                    contrasts.append((f'{f_cond}>{base_cond}', 'T', [f_cond, base_cond], [1, -1]))
                    contrasts.append((f'{f_cond}<{base_cond}', 'T', [f_cond, base_cond], [-1, 1]))
    
    logger.info(f"Generated {len(contrasts)} contrasts for {len(all_contrast_conditions)} conditions:")
    if cs_conditions['first']:
        logger.info(f"  - CS- first trial: {cs_conditions['first']} (separate condition)")
    if cs_conditions['other']:
        logger.info(f"  - CS- other trials: {cs_conditions['other']} (grouped as 'CS-_others')")
    if css_conditions['first']:
        logger.info(f"  - CSS first trial: {css_conditions['first']} (separate condition)")
    if css_conditions['other']:
        logger.info(f"  - CSS other trials: {css_conditions['other']} (grouped as 'CSS_others')")
    if csr_conditions['first']:
        logger.info(f"  - CSR first trial: {csr_conditions['first']} (separate condition)")
    if csr_conditions['other']:
        logger.info(f"  - CSR other trials: {csr_conditions['other']} (grouped as 'CSR_others')")
    logger.info(f"  - Other trial types: {other_conditions}")
    
    return contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions


def create_cs_separated_contrasts(df_trial_info, contrast_type='standard'):
    """
    Create contrasts with enhanced CS-, CSS, and CSR condition grouping.
    
    This function creates contrasts where:
    1. First trial of each condition type (CS-, CSS, CSR) is handled as a separate condition
    2. All other trials of each type are grouped together as single conditions
    3. All other trial types are handled as individual conditions
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                    The 'trial_type' column contains condition names,
                                    and 'onset' column is used for chronological sorting.
        contrast_type (str): Type of contrasts to create
    
    Returns:
        tuple: (contrasts_list, cs_conditions, css_conditions, csr_conditions, other_conditions, regressor_info)
            - contrasts_list: List of contrast tuples
            - cs_conditions: dict with 'first' and 'other' keys for CS- conditions
            - css_conditions: dict with 'first' and 'other' keys for CSS conditions
            - csr_conditions: dict with 'first' and 'other' keys for CSR conditions
            - other_conditions: List of non-CS/CSS/CSR conditions
            - regressor_info: Dictionary with regressor details for all condition types
    """
    contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_contrasts(df_trial_info, contrast_type)
    
    regressor_info = {
        'cs_conditions': cs_conditions,
        'css_conditions': css_conditions,
        'csr_conditions': csr_conditions,
        'other_conditions': other_conditions,
        'regressor_type': 'enhanced_grouping',
        'description': 'First trials kept separate, other trials grouped by type',
        'contrast_included': True,
        'grouping_strategy': 'first_trial_separate_others_grouped_by_type',
        'condition_names': {
            'cs_first': cs_conditions['first'],
            'cs_others': 'CS-_others' if cs_conditions['other'] else None,
            'css_first': css_conditions['first'],
            'css_others': 'CSS_others' if css_conditions['other'] else None,
            'csr_first': csr_conditions['first'],
            'csr_others': 'CSR_others' if csr_conditions['other'] else None,
            'other_types': other_conditions
        }
    }
    
    logger.info(f"Enhanced CS-/CSS/CSR condition grouping configured:")
    if cs_conditions['first']:
        logger.info(f"  - CS- first trial '{cs_conditions['first']}' kept as separate condition")
    if cs_conditions['other']:
        logger.info(f"  - CS- other trials grouped into 'CS-_others': {cs_conditions['other']}")
    if css_conditions['first']:
        logger.info(f"  - CSS first trial '{css_conditions['first']}' kept as separate condition")
    if css_conditions['other']:
        logger.info(f"  - CSS other trials grouped into 'CSS_others': {css_conditions['other']}")
    if csr_conditions['first']:
        logger.info(f"  - CSR first trial '{csr_conditions['first']}' kept as separate condition")
    if csr_conditions['other']:
        logger.info(f"  - CSR other trials grouped into 'CSR_others': {csr_conditions['other']}")
    
    return contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions, regressor_info


def create_custom_contrasts(df_trial_info, contrast_patterns):
    """
    Create custom contrasts based on specific patterns with enhanced CS-, CSS, and CSR grouping.
    
    This function implements the enhanced condition grouping strategy for CS-, CSS, and CSR conditions.
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                    The 'trial_type' column contains condition names,
                                    and 'onset' column is used for chronological sorting.
        contrast_patterns (list): List of contrast patterns or direct weight lists
    
    Returns:
        tuple: (contrasts_list, cs_conditions, css_conditions, csr_conditions, other_conditions)
            - contrasts_list: List of contrast tuples
            - cs_conditions: dict with 'first' and 'other' keys for CS- conditions
            - css_conditions: dict with 'first' and 'other' keys for CSS conditions
            - csr_conditions: dict with 'first' and 'other' keys for CSR conditions
            - other_conditions: List of non-CS/CSS/CSR conditions
    """
    if df_trial_info is None:
        raise ValueError("df_trial_info is required")
    
    if not contrast_patterns:
        raise ValueError("contrast_patterns is required")
    
    # Extract CS-, CSS, and CSR conditions with enhanced grouping
    df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions = extract_cs_conditions(df_trial_info)
    
    # Create list of all conditions for contrast generation
    all_contrast_conditions = []
    
    # Add first trials as separate conditions
    if cs_conditions['first']:
        all_contrast_conditions.append(cs_conditions['first'])
    if css_conditions['first']:
        all_contrast_conditions.append(css_conditions['first'])
    if csr_conditions['first']:
        all_contrast_conditions.append(csr_conditions['first'])
    
    # Add grouped other trials as single conditions
    if cs_conditions['other']:
        all_contrast_conditions.append('CS-_others')
    if css_conditions['other']:
        all_contrast_conditions.append('CSS_others')
    if csr_conditions['other']:
        all_contrast_conditions.append('CSR_others')
    
    # Add all other conditions
    all_contrast_conditions.extend(other_conditions)
    
    contrasts = []
    for pattern in contrast_patterns:
        if pattern == 'all_vs_baseline':
            for condition in all_contrast_conditions:
                contrasts.append((f'{condition}>baseline', 'T', [condition], [1]))
        
        elif pattern == 'pairwise':
            for i, cond1 in enumerate(all_contrast_conditions):
                for j, cond2 in enumerate(all_contrast_conditions):
                    if i < j:
                        contrasts.append((f'{cond1}>{cond2}', 'T', [cond1, cond2], [1, -1]))
                        contrasts.append((f'{cond1}<{cond2}', 'T', [cond1, cond2], [-1, 1]))
        
        elif pattern == 'first_vs_rest':
            if len(all_contrast_conditions) > 1:
                first_cond = all_contrast_conditions[0]
                rest_conds = all_contrast_conditions[1:]
                weights = [len(rest_conds)] + [-1] * len(rest_conds)
                contrasts.append((f'{first_cond}>rest', 'T', [first_cond] + rest_conds, weights))
        
        elif pattern == 'group_vs_group':
            # Split conditions into two groups
            mid = len(all_contrast_conditions) // 2
            group1 = all_contrast_conditions[:mid]
            group2 = all_contrast_conditions[mid:]
            if group1 and group2:
                weights = [1/len(group1)] * len(group1) + [-1/len(group2)] * len(group2)
                contrasts.append(('group1>group2', 'T', group1 + group2, weights))
        
        elif pattern == 'linear_trend':
            # Create linear trend contrast
            n_conds = len(all_contrast_conditions)
            if n_conds > 2:
                weights = [(i - (n_conds-1)/2) for i in range(n_conds)]
                contrasts.append(('linear_trend', 'T', all_contrast_conditions, weights))
    
    logger.info(f"Generated {len(contrasts)} custom contrasts with enhanced CS-/CSS/CSR grouping:")
    if cs_conditions['first']:
        logger.info(f"  - CS- first trial: {cs_conditions['first']} (separate condition)")
    if cs_conditions['other']:
        logger.info(f"  - CS- other trials: {cs_conditions['other']} (grouped as 'CS-_others')")
    if css_conditions['first']:
        logger.info(f"  - CSS first trial: {css_conditions['first']} (separate condition)")
    if css_conditions['other']:
        logger.info(f"  - CSS other trials: {css_conditions['other']} (grouped as 'CSS_others')")
    if csr_conditions['first']:
        logger.info(f"  - CSR first trial: {csr_conditions['first']} (separate condition)")
    if csr_conditions['other']:
        logger.info(f"  - CSR other trials: {csr_conditions['other']} (grouped as 'CSR_others')")
    logger.info(f"  - Other trial types: {other_conditions}")
    
    return contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions


def create_interesting_contrasts(df_trial_info):
    """
    Create only the interesting contrasts for NARSAD analysis.
    
    This function creates a focused set of contrasts that are most relevant
    for the NARSAD study, focusing on comparisons between "others" conditions
    and baseline, as well as between different "others" conditions.
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
    
    Returns:
        tuple: (contrasts_list, cs_conditions, css_conditions, csr_conditions, other_conditions)
    """
    if df_trial_info is None:
        raise ValueError("df_trial_info is required")
    
    # Extract CS-, CSS, and CSR conditions with grouping
    df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions = extract_cs_conditions(df_trial_info)
    
    # Use the conditions column for contrast generation
    all_contrast_conditions = df_with_conditions['conditions'].unique().tolist()
    
    # Define the interesting contrasts
    interesting_contrasts = [
        ("CS-_first_half_others > FIXATION_first_half", "first half Other CS- trials vs baseline"),
        ("CSS_first_half_others > FIXATION_first_half", "first half Other CSS trials vs baseline"),
        ("CSR_first_half_others > FIXATION_first_half", "first half Other CSR trials vs baseline"),
        ("CSS_first_half_others > CSR_first_half_others", "first half Other CSS trials vs Other CSR trials"),
        ("CSR_first_half_others > CSS_first_half_others", "first half Other CSR trials vs Other CSS trials"),
        ("CSS_first_half_others > CS-_first_half_others", "first half Other CSS trials vs Other CS- trials"),
        ("CSR_first_half_others > CS-_first_half_others", "first half Other CSR trials vs Other CS- trials"),
        ("CS-_first_half_others > CSS_first_half_others", "first half Other CS- trials vs Other CSS trials"),
        ("CS-_first_half_others > CSR_first_half_others", "first half Other CS- trials vs Other CSR trials"),
        ("CS-_second_half_others > FIXATION_second_half", "second half Other CS- trials vs baseline"),
        ("CSS_second_half_others > FIXATION_second_half ", "second half Other CSS trials vs baseline"),
        ("CSR_second_half_others > FIXATION_second_half", "second half Other CSR trials vs baseline"),
        ("CSS_second_half_others > CSR_second_half_others", "second half Other CSS trials vs Other CSR trials"),
        ("CSR_second_half_others > CSS_second_half_others", "second half Other CSR trials vs Other CSS trials"),
        ("CSS_second_half_others > CS-_second_half_others", "second half Other CSS trials vs Other CS- trials"),
        ("CSR_second_half_others > CS-_second_half_others", "second half Other CSR trials vs Other CS- trials"),
        ("CS-_second_half_others > CSS_second_half_others", "second half Other CS- trials vs Other CSS trials"),
        ("CS-_second_half_others > CSR_second_half_others", "second half Other CS- trials vs Other CSR trials"),
    ]
    
    contrasts = []
    
    for contrast_name, description in interesting_contrasts:
        # Parse the contrast name (e.g., "CS-_others > FIXATION")
        if ' > ' in contrast_name:
            condition1, condition2 = contrast_name.split(' > ')
            condition1 = condition1.strip()
            condition2 = condition2.strip()
            
            # Check if both conditions exist
            if condition1 in all_contrast_conditions and condition2 in all_contrast_conditions:
                contrast = (contrast_name, 'T', [condition1, condition2], [1, -1])
                contrasts.append(contrast)
                logger.info(f"Added contrast: {contrast_name} - {description}")
            else:
                logger.warning(f"Contrast {contrast_name}: conditions {condition1}, {condition2} not found in {all_contrast_conditions}")
        else:
            logger.warning(f"Invalid contrast format: {contrast_name}")
    
    logger.info(f"Created {len(contrasts)} interesting contrasts")
    
    return contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions

# =============================================================================
# CORE WORKFLOW FUNCTIONS
# =============================================================================

def first_level_wf(in_files, output_dir, df_trial_info, contrasts=None, 
                   contrast_type='interesting', contrast_patterns=None,
                   fwhm=6.0, brightness_threshold=1000, high_pass_cutoff=100,
                   use_smoothing=True, use_derivatives=True, model_serial_correlations=True):
    """
    Generic first-level workflow for fMRI analysis.
    
    Args:
        in_files (dict): Input files dictionary
        output_dir (str): Output directory path
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                    The 'trial_type' column contains condition names,
                                    and 'onset' column is used for chronological sorting.
        contrasts (list): List of contrast tuples (auto-generated if None)
        contrast_type (str): Type of contrasts to auto-generate ('standard', 'minimal', 'custom')
        contrast_patterns (list): List of contrast patterns for custom generation
        fwhm (float): Smoothing FWHM
        brightness_threshold (float): SUSAN brightness threshold
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured first-level workflow
    """
    if not in_files:
        raise ValueError("in_files cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    # Data source
    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
        function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')
    
    # Optional smoothing
    if use_smoothing:
        susan = pe.Node(SUSAN(), name='susan')
        susan.inputs.fwhm = fwhm
        susan.inputs.brightness_threshold = brightness_threshold
        preproc_output = susan
    else:
        preproc_output = apply_mask

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')

    # Auto-generate contrasts if not provided
    if contrasts is None:
        if condition_names is None:
            # Default condition names - can be overridden
            condition_names = ['condition1', 'condition2', 'condition3']
        
        if contrast_type == 'custom' and contrast_patterns:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_custom_contrasts(df_trial_info, contrast_patterns)
        else:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_contrasts(df_trial_info, contrast_type=contrast_type)
    
    if not contrasts:
        logger.warning("No contrasts generated, workflow may fail")
    
    logger.info(f"Using {len(contrasts)} contrasts: {[c[0] for c in contrasts]}")

    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'cope{i}'),
            name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'varcope{i}'),
            name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    # Build workflow connections
    connections = _build_workflow_connections(
        datasource, apply_mask, runinfo, l1_spec, l1_model, 
        feat_spec, feat_fit, feat_select, preproc_output, use_smoothing
    )
    
    # Add data sink connections
    for i in range(1, n_contrasts + 1):
        connections.extend([
            (datasource, ds_copes[i - 1], [('bold', 'source_file')]),
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')]),
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
        ])

    workflow.connect(connections)
    return workflow

def first_level_wf_LSS(in_files, output_dir, trial_ID, df_trial_info, contrasts=None,
                       contrast_type='minimal', contrast_patterns=None,
                       fwhm=6.0, brightness_threshold=1000, high_pass_cutoff=100,
                       use_smoothing=False, use_derivatives=True, model_serial_correlations=True):
    """
    Generic LSS (Least Squares Separate) first-level workflow.
    
    Note: LSS analysis is recommended to be run WITHOUT smoothing to preserve
    fine-grained temporal information and avoid blurring trial-specific responses.
    
    Args:
        in_files (dict): Input files dictionary
        output_dir (str): Output directory path
        trial_ID (int): Trial ID for LSS analysis
        condition_names (list): List of condition names (auto-detected if None)
        contrasts (list): List of contrast tuples (auto-generated if None)
        contrast_type (str): Type of contrasts to auto-generate ('minimal', 'standard', 'custom')
        contrast_patterns (list): List of contrast patterns for custom generation
        fwhm (float): Smoothing FWHM (not recommended for LSS)
        brightness_threshold (float): SUSAN brightness threshold (not used if use_smoothing=False)
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing (default: False for LSS)
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured LSS workflow
    """
    if not in_files:
        raise ValueError("in_files cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level_LSS')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    datasource = pe.Node(niu.Function(function=_dict_ds_lss, output_names=DATA_ITEMS_LSS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file',
                     'trial_ID', 'regressors_names', 'motion_columns',
                     'decimals', 'amplitude'],
        output_names=['info', 'realign_file'],
        function=_bids2nipypeinfo_lss),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')
    
    # Note: LSS typically does not use smoothing to preserve temporal precision
    if use_smoothing:
        logger.warning("Smoothing is enabled for LSS analysis. This is not recommended as it may blur trial-specific responses.")

    # Auto-generate contrasts if not provided
    if contrasts is None:
        if condition_names is None:
            # Default LSS contrasts
            condition_names = ['trial', 'others']
        
        if contrast_type == 'custom' and contrast_patterns:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_custom_contrasts(df_trial_info, contrast_patterns)
        else:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_contrasts(df_trial_info, contrast_type=contrast_type)
    
    if not contrasts:
        logger.warning("No contrasts generated for LSS workflow")
    
    logger.info(f"LSS using {len(contrasts)} contrasts: {[c[0] for c in contrasts]}")

    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_cope{i}'),
            name=f'ds_cope{i}',
            run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_varcope{i}'),
            name=f'ds_varcope{i}',
            run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    # Workflow connections
    connections = [
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, runinfo, [('events', 'events_file'), ('trial_ID', 'trial_ID'), ('regressors', 'regressors_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
        (apply_mask, runinfo, [('out_file', 'in_file')]),
        (runinfo, l1_spec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (apply_mask, feat_fit, [('out_file', 'in_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
    ]
    
    # Add data sink connections
    for i in range(1, n_contrasts + 1):
        connections.extend([
            (datasource, ds_copes[i - 1], [('bold', 'source_file')]),
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')]),
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
        ])

    workflow.connect(connections)
    return workflow

def first_level_wf_voxelwise(inputs, output_dir, df_trial_info, contrasts=None, 
                            contrast_type='interesting', contrast_patterns=None, fwhm=6.0, 
                            brightness_threshold=0.1, high_pass_cutoff=128, use_smoothing=True, 
                            use_derivatives=True, model_serial_correlations=True):
    """
    Specialized first-level workflow for voxel-wise analysis with CS- condition handling.
    
    This workflow is specifically designed for voxel-wise analysis where CS- conditions
    need to be handled as separate regressors, separate from the main condition contrasts.
    
    Args:
        inputs (dict): Input files dictionary
        output_dir (str): Output directory path
        condition_names (list): List of condition names (auto-detected if None)
        contrasts (list): List of contrast tuples (auto-generated if None)
        contrast_type (str): Type of contrasts to auto-generate ('standard', 'minimal', 'custom')
        contrast_patterns (list): List of contrast patterns for custom generation
        fwhm (float): Smoothing FWHM
        brightness_threshold (float): SUSAN brightness threshold
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured first-level workflow for voxel-wise analysis
    """
    if not inputs:
        raise ValueError("inputs cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level_voxelwise')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    # Data source
    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = inputs
    datasource.iterables = ('sub', sorted(inputs.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
        function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')
    
    # Optional smoothing
    if use_smoothing:
        susan = pe.Node(SUSAN(), name='susan')
        susan.inputs.fwhm = fwhm
        susan.inputs.brightness_threshold = brightness_threshold
        preproc_output = susan
    else:
        preproc_output = apply_mask

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')

    # Auto-generate contrasts if not provided
    if contrasts is None:
        if condition_names is None:
            # Default condition names - can be overridden
            condition_names = ['condition1', 'condition2', 'condition3']
        
        if contrast_type == 'custom' and contrast_patterns:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_custom_contrasts(df_trial_info, contrast_patterns)
        else:
            contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_contrasts(df_trial_info, contrast_type=contrast_type)
    
    if not contrasts:
        logger.warning("No contrasts generated, workflow may fail")
    
    # Log first trial of CS-, CSS, and CSR condition handling
    if cs_conditions['first'] or css_conditions['first'] or csr_conditions['first']:
        logger.info(f"First trials will be handled as separate regressors:")
        if cs_conditions['first']:
            logger.info(f"  - CS- first trial: '{cs_conditions['first']}'")
        if css_conditions['first']:
            logger.info(f"  - CSS first trial: '{css_conditions['first']}'")
        if csr_conditions['first']:
            logger.info(f"  - CSR first trial: '{csr_conditions['first']}'")
        logger.info(f"Main contrasts: {[c[0] for c in contrasts]}")
    else:
        logger.info(f"No first trials found. Using {len(contrasts)} contrasts: {[c[0] for c in contrasts]}")

    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=output_dir,
            suffix=f'cope{i}',
            desc='preproc',
            compress=True
        ), name=f'ds_cope{i}')
        for i in range(1, n_contrasts + 1)
    ]
    
    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=output_dir,
            suffix=f'varcope{i}',
            desc='preproc',
            compress=True
        ), name=f'ds_varcope{i}')
        for i in range(1, n_contrasts + 1)
    ]

    # Connect the workflow
    workflow.connect([
        (datasource, runinfo, [('bold', 'in_file')]),
        (datasource, runinfo, [('events', 'events_file')]),
        (datasource, runinfo, [('regressors', 'regressors_file')]),
        (datasource, apply_mask, [('mask', 'mask_file')]),
        (datasource, preproc_output, [('bold', 'in_file')]),
        (runinfo, l1_spec, [('info', 'subject_info')]),
        (runinfo, l1_spec, [('realign_file', 'realignment_parameters')]),
        (preproc_output, l1_spec, [('out_file', 'functional_runs')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file')]),
        (feat_spec, feat_fit, [('fsf_file', 'design_file')]),
        (preproc_output, feat_fit, [('out_file', 'in_file')]),
        (feat_fit, feat_select, [('copes', 'copes')]),
        (feat_fit, feat_select, [('varcopes', 'varcopes')]),
    ])

    # Connect data sinks
    for i, (ds_cope, ds_varcope) in enumerate(zip(ds_copes, ds_varcopes), 1):
        workflow.connect([
            (feat_select, ds_cope, [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcope, [(f'varcope{i}', 'in_file')])
        ])

    return workflow

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_workflow_connections(datasource, apply_mask, runinfo, l1_spec, l1_model, 
                              feat_spec, feat_fit, feat_select, preproc_output, use_smoothing):
    """
    Build workflow connections based on smoothing configuration.
    
    Args:
        datasource: Data source node
        apply_mask: Mask application node
        runinfo: Run info node
        l1_spec: Level 1 specification node
        l1_model: Level 1 model node
        feat_spec: FEAT specification node
        feat_fit: FEAT fitting node
        feat_select: FEAT selection node
        preproc_output: Preprocessing output node
        use_smoothing: Whether smoothing is used
    
    Returns:
        list: List of workflow connections
    """
    connections = [
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, runinfo, [('events', 'events_file'), ('regressors', 'regressors_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
    ]
    
    # Add smoothing connections if used
    if use_smoothing:
        connections.extend([
            (apply_mask, preproc_output, [('out_file', 'in_file')]),
            (preproc_output, l1_spec, [('smoothed_file', 'functional_runs')]),
            (preproc_output, runinfo, [('smoothed_file', 'in_file')]),
            (preproc_output, feat_fit, [('smoothed_file', 'in_file')])
        ])
    else:
        connections.extend([
            (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
            (apply_mask, runinfo, [('out_file', 'in_file')]),
            (apply_mask, feat_fit, [('out_file', 'in_file')])
        ])
    
    # Add runinfo connections
    connections.extend([
        (runinfo, l1_spec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')])
    ])
    
    return connections

def create_voxelwise_design_matrix(df_trial_info):
    """
    Create a design matrix specifically for voxel-wise analysis with enhanced CS-, CSS, and CSR condition grouping.
    
    This function creates a design matrix where:
    1. First trial of each condition type (CS-, CSS, CSR) is handled as a separate condition
    2. All other trials of each type are grouped together as single conditions
    3. All other trial types are handled as individual conditions
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                    The 'trial_type' column contains condition names,
                                    and 'onset' column is used for chronological sorting.
    
    Returns:
        dict: Design matrix configuration with enhanced condition grouping
    """
    df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions = extract_cs_conditions(df_trial_info)
    
    design_config = {
        'main_conditions': other_conditions,
        'cs_conditions': cs_conditions,
        'css_conditions': css_conditions,
        'csr_conditions': csr_conditions,
        'enhanced_grouping': True,
        'description': 'Voxel-wise analysis with enhanced CS-/CSS/CSR condition grouping: first trials separate, others grouped by type'
    }
    
    logger.info(f"Design matrix configuration:")
    if cs_conditions['first']:
        logger.info(f"  - CS- first trial '{cs_conditions['first']}' as separate condition")
    if cs_conditions['other']:
        logger.info(f"  - CS- other trials grouped as 'CS-_others': {cs_conditions['other']}")
    if css_conditions['first']:
        logger.info(f"  - CSS first trial '{css_conditions['first']}' as separate condition")
    if css_conditions['other']:
        logger.info(f"  - CSS other trials grouped as 'CSS_others': {css_conditions['other']}")
    if csr_conditions['first']:
        logger.info(f"  - CSR first trial '{csr_conditions['first']}' as separate condition")
    if csr_conditions['other']:
        logger.info(f"  - CSR other trials grouped as 'CSR_others': {csr_conditions['other']}")
    logger.info(f"  - Other conditions: {other_conditions}")
    
    return design_config

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_standard_contrasts():
    """Create standard contrasts for typical fMRI analysis."""
    condition_names = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    return create_contrasts(condition_names, contrast_type='standard')

def create_lss_contrasts():
    """Create standard LSS contrasts."""
    condition_names = ['trial', 'others']
    return create_contrasts(condition_names, contrast_type='minimal')

def create_face_house_contrasts():
    """Create contrasts for face vs house experiment."""
    condition_names = ['face', 'house', 'object']
    return create_contrasts(condition_names, contrast_type='standard')

def create_emotion_contrasts():
    """Create contrasts for emotion experiment."""
    condition_names = ['happy', 'sad', 'angry', 'neutral']
    return create_contrasts(condition_names, contrast_type='standard')

def create_working_memory_contrasts():
    """Create contrasts for working memory experiment."""
    condition_names = ['load1', 'load2', 'load3', 'load4']
    patterns = ['all_vs_baseline', 'linear_trend', 'quadratic_trend']
    return create_custom_contrasts(condition_names, patterns)

# =============================================================================
# BEST PRACTICES AND DOCUMENTATION
# =============================================================================

def create_lss_workflow_best_practices():
    """
    LSS (Least Squares Separate) Analysis Best Practices:
    
    1. NO SMOOTHING: LSS should typically be run without spatial smoothing
       to preserve fine-grained temporal information and avoid blurring
       trial-specific responses.
    
    2. TEMPORAL PRECISION: LSS is designed to capture trial-by-trial
       variability, so temporal precision is crucial.
    
    3. MINIMAL CONTRASTS: Usually only need trial vs baseline or trial vs others.
    
    4. HIGH-PASS FILTERING: Use appropriate high-pass filtering to remove
       low-frequency drifts while preserving trial-specific signals.
    
    Example:
        wf = first_level_wf_LSS(
            in_files=files, output_dir='output', trial_ID=1,
            use_smoothing=False,  # CRITICAL for LSS
            high_pass_cutoff=100,  # Adjust based on your design
            use_derivatives=False  # Often not needed for LSS
        )
    """
    pass

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """
    Example usage of the first-level workflows with enhanced CS- condition grouping.
    
    This demonstrates how to use the workflows for voxel-wise analysis where:
    1. First trial of CS- conditions is handled as a separate condition (keeps original name)
    2. All other CS- trials are kept as individual conditions (keeps original names)
    3. All other trial types are handled as individual conditions
    """
    # Example 1: Enhanced CS- condition grouping for voxel-wise analysis
    print("=== Example 1: Enhanced CS- condition grouping for voxel-wise analysis ===")
    
    # Sample condition names (first trial of CS- condition will be auto-detected)
    condition_names = ['CS-US_trial1', 'CS-US_trial2', 'CS-US_trial3', 'US', 'CS+', 'baseline']
    
    # Create contrasts with enhanced CS-, CSS, and CSR grouping
    contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions = create_contrasts(condition_names, 'standard')
    
    print(f"CS- conditions: first='{cs_conditions['first']}', other={cs_conditions['other']}")
    print(f"CSS conditions: first='{css_conditions['first']}', other={css_conditions['other']}")
    print(f"CSR conditions: first='{csr_conditions['first']}', other={csr_conditions['other']}")
    print(f"Other trial types: {other_conditions}")
    print(f"Generated contrasts: {[c[0] for c in contrasts]}")
    
    # Example 2: Voxel-wise workflow with enhanced CS- grouping
    print("\n=== Example 2: Voxel-wise workflow setup with enhanced grouping ===")
    
    # Create design matrix configuration
    design_config = create_voxelwise_design_matrix(condition_names)
    print(f"Design matrix config: {design_config}")
    
    # Example 3: Custom contrast patterns with enhanced CS- grouping
    print("\n=== Example 3: Custom contrasts with enhanced CS- grouping ===")
    
    custom_patterns = ['all_vs_baseline', 'pairwise']
    contrasts, cs_first_trial, cs_other_trials, other_conditions = create_custom_contrasts(condition_names, custom_patterns)
    
    print(f"Custom contrasts generated: {[c[0] for c in contrasts]}")
    
    # Example 4: Enhanced CS- separated contrasts
    print("\n=== Example 4: Enhanced CS- separated contrast creation ===")
    
    contrasts, cs_first_trial, cs_other_trials, other_conditions, cs_regressor_info = create_cs_separated_contrasts(condition_names)
    print(f"Enhanced CS- regressor info: {cs_regressor_info}")
    
    # Example 5: Different trial naming conventions with enhanced grouping
    print("\n=== Example 5: Different trial naming conventions with enhanced grouping ===")
    
    # Test different naming formats
    test_conditions = ['CS-US_1', 'CS-US_2', 'CS-US_3', 'US', 'CS+', 'baseline']
    contrasts, cs_first_trial, cs_other_trials, other_conditions = create_contrasts(test_conditions, 'standard')
    print(f"Format 'CS-US_1': First trial = {cs_first_trial}, Other CS- trials grouped into 'CS-_others': {cs_other_trials}")
    
    test_conditions_2 = ['CS-US_trial1', 'CS-US_trial2', 'US', 'baseline']
    contrasts, cs_first_trial, cs_other_trials, other_conditions = create_contrasts(test_conditions_2, 'standard')
    print(f"Format 'CS-US_trial1': First trial = {cs_first_trial}, Other CS- trials grouped into 'CS-_others': {cs_other_trials}")
    
    test_conditions_3 = ['CS-US', 'US', 'baseline']  # Single CS- condition
    contrasts, cs_first_trial, cs_other_trials, other_conditions = create_contrasts(test_conditions_2, 'standard')
    print(f"Format 'CS-US': First trial = {cs_first_trial}, Other CS- trials grouped into 'CS-_others': {cs_other_trials}")
    
    # Example 6: Real-world event file example with new naming convention
    print("\n=== Example 6: Real-world event file example with new naming convention ===")
    
    # Based on your event file with CS-, CSS, CSR, US conditions
    real_conditions = ['CS-', 'CSS', 'US_CSS', 'CSR', 'US_CSR', 'FIXATION']
    contrasts, cs_first_trial, cs_other_trials, other_conditions = create_contrasts(real_conditions, 'standard')
    
    print(f"Real event file conditions: {real_conditions}")
    print(f"First trial of CS- condition: {cs_first_trial}")
    print(f"Other CS- trials grouped into 'CS-_others': {cs_other_trials}")
    print(f"Other trial types: {other_conditions}")
    print(f"Final condition names for contrasts: {cs_first_trial}, 'CS-_others', {other_conditions}")
    print(f"Total contrasts generated: {len(contrasts)}")
    
    print("\n=== Summary ===")
    print("The workflows now automatically implement ENHANCED CS- condition grouping:")
    print("1. Detect the FIRST TRIAL of CS- conditions and keep as separate condition")
    print("2. Group ALL OTHER CS- trials into single 'CS-_others' condition")
    print("3. Handle ALL OTHER trial types as INDIVIDUAL conditions")
    print("4. Support multiple trial naming conventions:")
    print("   - CS-US_1, CS-US_2, CS-US_3")
    print("   - CS-US_trial1, CS-US_trial2, CS-US_trial3")
    print("   - CS-US (single condition)")
    print("5. Create contrasts between:")
    print("   - First CS- trial vs grouped CS- trials vs other conditions")
    print("   - Grouped CS- trials vs other conditions")
    print("   - All other trial types vs each other")
    print("6. Provide enhanced design matrix configuration for voxel-wise analysis")
    print("7. Support comprehensive logging of enhanced CS- grouping strategy")
    print("8. Enable sophisticated contrast analysis with proper CS- trial separation")
    print("9. Use original condition names to ensure FSL compatibility")
