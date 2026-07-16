
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, output_dim):
        super(CrossAttention, self).__init__()        
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_input, key_input, value_input, query_input_mask, key_input_mask, attn_bias=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(value_input)

        attn_weights = torch.matmul(query, key.transpose(1, 2)) / self.scale_val
        
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        
        attn_mask = query_input_mask.unsqueeze(-1) * key_input_mask.unsqueeze(-1).transpose(1, 2)
        attn_weights = attn_weights.masked_fill(attn_mask == False, -1e9)
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class EnzymeCompoundCrossAttention(nn.Module):
    def __init__(self, enz_node_dim, cpd_node_dim, output_dim, use_prods_info):
        super(EnzymeCompoundCrossAttention, self).__init__()
        self.cross_attn_enzyme = CrossAttention(enz_node_dim, cpd_node_dim, output_dim)
        self.cross_attn_substrate = CrossAttention(cpd_node_dim, enz_node_dim, output_dim)
        self.cross_attn_product = CrossAttention(cpd_node_dim, enz_node_dim, output_dim)
        self.use_prods_info = use_prods_info
    
    def forward(
        self, 
        enz_node_feature, 
        substrate_node_feature, 
        product_node_feature, 
        enz_node_feature_mask,
        substrate_node_feature_mask,
        product_node_feature_mask,
        interaction_weight=None,
        return_weights=False,
    ):
        
        if interaction_weight is not None:
            subs_pocket_weight = interaction_weight
            pocket_subs_weight = interaction_weight.transpose(1, 2)
        else:
            subs_pocket_weight, pocket_subs_weight = None, None
        
        # Enzyme-Substrate
        enzyme_subs_output, enzyme_subs_weights = self.cross_attn_enzyme(query_input=enz_node_feature, 
                                                       key_input=substrate_node_feature, 
                                                       value_input=substrate_node_feature, 
                                                       query_input_mask=enz_node_feature_mask, 
                                                       key_input_mask=substrate_node_feature_mask,
                                                       attn_bias=pocket_subs_weight)

        subs_enzyme_output, _ = self.cross_attn_substrate(query_input=substrate_node_feature, 
                                                          key_input=enz_node_feature, 
                                                          value_input=enz_node_feature, 
                                                          query_input_mask=substrate_node_feature_mask, 
                                                          key_input_mask=enz_node_feature_mask,
                                                          attn_bias=subs_pocket_weight)

        if self.use_prods_info:
            # Enzyme-Product
            enzyme_prod_output, _ = self.cross_attn_enzyme(query_input=enz_node_feature, 
                                                        key_input=product_node_feature, 
                                                        value_input=product_node_feature, 
                                                        query_input_mask=enz_node_feature_mask, 
                                                        key_input_mask=product_node_feature_mask)

            prod_enzyme_output, _ = self.cross_attn_product(query_input=product_node_feature, 
                                                            key_input=enz_node_feature, 
                                                            value_input=enz_node_feature, 
                                                            query_input_mask=product_node_feature_mask, 
                                                            key_input_mask=enz_node_feature_mask)

            cross_attn_output = torch.cat([
                    enzyme_subs_output.mean(1), 
                    subs_enzyme_output.mean(1),
                    enzyme_prod_output.mean(1),
                    prod_enzyme_output.mean(1)
                ], 
                dim=-1)
        else:
            cross_attn_output = torch.cat([enzyme_subs_output.mean(1), subs_enzyme_output.mean(1)], dim=-1)
        
        if return_weights:
            return cross_attn_output, enzyme_subs_weights
        else:
            return cross_attn_output

        