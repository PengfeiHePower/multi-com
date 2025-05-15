import os
# from llm_loading import load_model_and_tokenizer
import torch
import json
import joblib
import numpy as np

from fact_util import get_fact_flag

# load model and tokenizer
# model, tokenizer = load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

# TODO: wrap the model loading into a function, for different models
# load regression model, default is llama3_8B
def load_trust_models(model_type="llama3"):
    if model_type == "llama3":
        directory = "/egr/research-dselab/hepengf1/Documents/multi-com/attentionS/regression_saver/"

    clf_fact = joblib.load(directory + "trust_model_fact.pkl")
    scaler_fact = joblib.load(directory + "scaler_fact.pkl")
    le_fact = joblib.load(directory + "label_encoder_fact.pkl")

    clf_logic = joblib.load(directory + "trust_model_logic.pkl")
    scaler_logic = joblib.load(directory + "scaler_logic.pkl")
    le_logic = joblib.load(directory + "label_encoder_logic.pkl")

    clf_revelant = joblib.load(directory + "trust_model_revelant.pkl")
    scaler_revelant = joblib.load(directory + "scaler_revelant.pkl")
    le_revelant = joblib.load(directory + "label_encoder_revelant.pkl")

    clf_bias = joblib.load(directory + "trust_model_bias.pkl")
    scaler_bias = joblib.load(directory + "scaler_bias.pkl")
    le_bias = joblib.load(directory + "label_encoder_bias.pkl")

    clf_quality = joblib.load(directory + "trust_model_quality.pkl")
    scaler_quality = joblib.load(directory + "scaler_quality.pkl")
    le_quality = joblib.load(directory + "label_encoder_quality.pkl")

    clf_clarity = joblib.load(directory + "trust_model_clarity.pkl")
    scaler_clarity = joblib.load(directory + "scaler_clarity.pkl")
    le_clarity = joblib.load(directory + "label_encoder_clarity.pkl")

    rubric_models = {
        "bias": clf_bias,
        "clarity": clf_clarity,
        "fact": clf_fact,
        "logic": clf_logic,
        "quality": clf_quality,
        "revelant": clf_revelant,
    }

    rubric_scalers = {
        "bias": scaler_bias,
        "clarity": scaler_clarity,
        "fact": scaler_fact,
        "logic": scaler_logic,
        "quality": scaler_quality,
        "revelant": scaler_revelant,
    }

    rubric_le = {
        "bias": le_bias,
        "clarity": le_clarity,
        "fact": le_fact,
        "logic": le_logic,
        "quality": le_quality,
        "revelant": le_revelant,
    }

    return rubric_models, rubric_scalers, rubric_le

# load thresholds
def load_thresholds(file_path):
    """
    load thresholds from a json file
    """
    with open(file_path, 'r') as f:
        thresholds = json.load(f)
    return thresholds

# extratc attention scores from model
model_extract_config = {
    "llama3": {"user_start":4, "user_end":-1},
    "qwen2.5": {"user_start":3, "user_end":-2},
    "gemma3": {"user_start":-1, "user_end":-2}
}

layer_config = {
    "llama3": {"layer_num":15},
    "qwen2.5": {"layer_num":12},
    "gemma3": {"layer_num":15}
}

def geometric_mean(tensor):
    """
    return the geometric mean of the tensor
    """
    log_tensor = torch.log(tensor)
    geometric_mean = torch.exp(torch.mean(log_tensor))
    return geometric_mean

def get_attentions(model,tokenizer,model_type, system_prompt, user_prompt):
    """
    return a aggregated attention matrix in the shape of [num_layers, num_heads]
    use geometric mean to aggregate the attention scores
    obtain the attention score only on the user prompt
    """
    if model_type in ["llama3", "qwen2.5"]:
        messages = [
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs_assistant = tokenizer.apply_chat_template([
            {"role": "assistant",
             "content": system_prompt
             }], return_tensors="pt")
    elif model_type == "gemma3":
        inputs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt}
                        ]
                        }
                        ]
        inputs = tokenizer.apply_chat_template(inputs, return_tensors="pt")
        inputs_assistant = tokenizer.apply_chat_template([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt}
                    ]
                    }], return_tensors="pt")

    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(inputs)
    attentions = outputs.attentions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    #include system prompt
    # geo_head_layer_mat_full = torch.zeros([num_layers,num_heads])
    # for layer_num in range(num_layers):
    #     for attention_head_num in range(num_heads):
    #         attention=attentions[layer_num][0, attention_head_num].cpu()
    #         geo_head_layer_mat_full[layer_num, attention_head_num]=geometric_mean(attention[-1,:-1])
    
    #only user_prompts
    user_start = len(inputs_assistant[0])
    geo_head_layer_mat_user = torch.zeros([num_layers,num_heads])
    for layer_num in range(num_layers):
        for attention_head_num in range(num_heads):
            attention=attentions[layer_num][0, attention_head_num].cpu()
            geo_head_layer_mat_user[layer_num, attention_head_num]=geometric_mean(attention[-1,user_start+model_extract_config[model_type]["user_start"]:model_extract_config[model_type]["user_end"]])
    
    # return geo_head_layer_mat_full, geo_head_layer_mat_user
    return geo_head_layer_mat_user

def extract_trust_vector(attn_matrix, model_type):
    """
    extratc the trust vector from the the attention matrix
    return a vector in the shape of [num_heads]
    args:
        attn_matrix (torch.Tensor): the attention matrix
        layer_num (int): which layers to extract the trust vector from, default is 15
    return:
        a numpy array of trust scores in the shape of [num_heads]
    """
    layer_num = layer_config[model_type]['layer_num']
    head_scores = attn_matrix[1:layer_num,].mean(axis=0)
    head_scores = head_scores.numpy()
    return  head_scores

def trust_scores_per(attn_vector, rubric_models, rubric_scalers,rubric_le, mask=None,return_prob=True):
    """
    Compute trust scores for a single attention vector using per-rubric regression models and scalers.

    Args:
        attn_vector (np.ndarray): A 1D array of shape (num_heads,)
        rubric_models (dict): Dictionary of {rubric_name: fitted_model}
        rubric_scalers (dict): Dictionary of {rubric_name: fitted_scaler}
        return_prob (bool): If True, return probabilities; else, return raw logits.
        mask (np.ndarray): the mask of the user prompt,
    Returns:
        dict: {rubric_name: trust_score}
    """
    trust_scores = {}
    
    if mask is not None:
        attn_vector = attn_vector * mask
    for rubric in rubric_models:
        model = rubric_models[rubric]
        scaler = rubric_scalers[rubric]
        le = rubric_le[rubric]

        # Scale the input vector
        vec_scaled = scaler.transform(attn_vector.reshape(1, -1))

        # Get prediction: either logit or probability
        if return_prob:
            classes = list(le.classes_)
            if rubric not in classes:
                raise ValueError(f"Rubric '{rubric}' not found in model.classes_: {classes}")
            idx = classes.index(rubric)
            score = model.predict_proba(vec_scaled)[0][idx] # Prob of violation
        else:
            score = model.decision_function(vec_scaled)[0]  # Raw logit

        trust_scores[rubric] = score

    return trust_scores

def trust_violation_flag(trust_scores, thresholds):
    """
    return a flag indicating whether the trust scores are violated
    """
    flags = {'bias':False, 'clarity':False, 'fact':False, 'logic':False, 'quality':False, 'revelant':False}
    for rubric in trust_scores:
        if trust_scores[rubric] > thresholds['normal'][rubric]:
            flags[rubric] = True
    flags['overall'] = any(flags.values())
    return flags

def trust_violation_flag_double(trust_scores, thresholds, context, messages):
    """
    return a flag indicating whether the trust scores are violated
    """
    flags = {'bias':False, 'clarity':False, 'fact':False, 'logic':False, 'quality':False, 'revelant':False}
    single_rubric = ['logic', 'quality', 'revelant', 'bias', 'clarity']
    for rubric in single_rubric:
        if trust_scores[rubric] > thresholds['normal'][rubric]:
            flags[rubric] = True
    if trust_scores['fact'] > thresholds['normal']['fact'][1]:
        flags['fact'] = True
    elif trust_scores['fact'] < thresholds['normal']['fact'][0]:
        flags['fact'] = False
    elif trust_scores['fact'] >= thresholds['normal']['fact'][0] and trust_scores['fact'] <= thresholds['normal']['fact'][1]:
        flags['fact'] = get_fact_flag(context, messages)
    flags['overall'] = any(flags.values())
    return flags