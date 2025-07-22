import torch
new_model_dict = dict(torch.load('R-GWAE_10_1919_30_mmd_1_2_1000_mu-500_si-500_Channels_1-3.pt'))
keys = list(new_model_dict.keys())
for key in keys:
    print(key)
    if key.startswith('encoder'):
        new_key = '.'.join(key.split('.')[1:])
        if new_key.startswith('q_mu'):        
            new_key = new_key.replace('q_mu', 'q_mu.0')    
        if new_key.startswith('q_var'):
            new_key = new_key.replace('q_var', 'q_t')            
        new_model_dict[new_key] = new_model_dict[key]
        del new_model_dict[key]
    if key.startswith('decoder.decoder'):
        new_key = key.replace('decoder.decoder', 'p_mu')
        new_model_dict[new_key] = new_model_dict[key]
        del new_model_dict[key]
    if key.startswith('decoder.p_mu'):
        new_key = key.replace('decoder.', '').replace('0','4').replace('2','6')
        new_model_dict[new_key] = new_model_dict[key]
        del new_model_dict[key]
    if key.startswith('decoder.p_sigma'):
        new_key = key.replace('decoder.', '')
        new_model_dict[new_key] = new_model_dict[key]
        del new_model_dict[key]
        
torch.save(new_model_dict, 'rvae.pt')

new_state_dict = torch.load('rvae.pt')
trainer.model.load_state_dict(new_model_dict, strict=False)