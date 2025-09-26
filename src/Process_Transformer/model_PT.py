import torch
import torch.nn as nn
import math
from src.utils.fds import FDS


def get_PT_model(args, cfg, device=None):
    # define model parameters
    import pickle
    with open(args.vocab_size_path, 'rb') as f:
        vocab_size = pickle.load(f)
    with open(args.max_len_path, 'rb') as f:
        max_len = pickle.load(f)
    # infer time feature dimension from saved tensors
    try:
        import torch
        x_time = torch.load(args.Xtime_train_path, weights_only=True)
        time_dim = x_time.shape[1]
    except Exception:
        time_dim = 3
    embed_dim = cfg['PT']['embed_dim'] if 'PT' in cfg else 36
    num_heads = cfg['PT']['num_heads'] if 'PT' in cfg else 4
    ff_dim = cfg['PT']['ff_dim'] if 'PT' in cfg else 64
    feature_dim = 128
    fds_config = dict(
        feature_dim=feature_dim, start_update=args.fds_start_update,
        start_smooth=args.fds_start_smooth, kernel=args.fds_kernel,
        ks=args.fds_ks, sigma=args.fds_sigma)
    # select variant similar to DALSTM
    if args.bmse and args.FDS:
        model = ProcessTransformerFDSMVE(max_len, vocab_size, embed_dim, num_heads, ff_dim, time_dim=time_dim)
    elif args.FDS:
        model = ProcessTransformerFDS(max_len, vocab_size, embed_dim, num_heads, ff_dim, time_dim=time_dim)
    elif args.heteroscedastic or args.bmse:
        model = ProcessTransformerMVE(max_len, vocab_size, embed_dim, num_heads, ff_dim, time_dim=time_dim)
    else:
        model = ProcessTransformer(max_len, vocab_size, embed_dim, num_heads, ff_dim, time_dim=time_dim)
    if device is not None:
        model = model.to(device)
    return model, fds_config
	

##### Helper Classes for Process Transformer #####
class TransformerBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
		super(TransformerBlock, self).__init__()
		self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		self.ffn = nn.Sequential(
			nn.Linear(embed_dim, ff_dim),
			nn.ReLU(),
			nn.Linear(ff_dim, embed_dim),
		)
		self.layernorm_a = nn.LayerNorm(embed_dim, eps=1e-6)
		self.layernorm_b = nn.LayerNorm(embed_dim, eps=1e-6)
		self.dropout_a = nn.Dropout(rate)
		self.dropout_b = nn.Dropout(rate)

	def forward(self, inputs):
		attn_output, _ = self.att(inputs, inputs, inputs, need_weights=False)
		attn_output = self.dropout_a(attn_output)
		out_a = self.layernorm_a(inputs + attn_output)
		ffn_output = self.ffn(out_a)
		ffn_output = self.dropout_b(ffn_output)
		return self.layernorm_b(out_a + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
	def __init__(self, maxlen, vocab_size, embed_dim):
		super(TokenAndPositionEmbedding, self).__init__()
		self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
		self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embed_dim)

	def forward(self, x):
		batch_size, seq_len = x.size(0), x.size(1)
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		pos_embeddings = self.pos_emb(positions)
		token_embeddings = self.token_emb(x)
		return token_embeddings + pos_embeddings


##### Process Transformer #####
class PTBackbone(nn.Module):
    def __init__(self, max_case_length, vocab_size, embed_dim=36, num_heads=4, ff_dim=64, time_dim=3):
        super(PTBackbone, self).__init__()
        self.token_pos = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.time_dense = nn.Linear(time_dim, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(embed_dim + 32, 128)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward_features(self, token_inputs, time_inputs):
        x = self.token_pos(token_inputs)
        x = self.transformer_block(x)
        x = x.mean(dim=1) # Pooling
        x_t = torch.relu(self.time_dense(time_inputs)) # time inputs
        x = torch.cat([x, x_t], dim=1) # concatenate categorical and temporal features
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        features = self.dropout2(x)
        return features


class ProcessTransformer(nn.Module):
    def __init__(self, max_case_length, vocab_size, embed_dim=36, num_heads=4, ff_dim=64, time_dim=3, return_squeezed=True):
        super(ProcessTransformer, self).__init__()
        self.return_squeezed = return_squeezed
        self.backbone = PTBackbone(max_case_length, vocab_size, embed_dim, num_heads, ff_dim, time_dim)
        self.out = nn.Linear(128, 1)

    def forward(self, token_inputs, time_inputs):
        features = self.backbone.forward_features(token_inputs, time_inputs)
        outputs = self.out(features)
        if self.return_squeezed:
            return outputs.squeeze(dim=1)
        return outputs


class ProcessTransformerMVE(nn.Module):
    def __init__(self, max_case_length, vocab_size, embed_dim=36, num_heads=4, ff_dim=64, time_dim=3, return_squeezed=True):
        super(ProcessTransformerMVE, self).__init__()
        self.return_squeezed = return_squeezed
        self.backbone = PTBackbone(max_case_length, vocab_size, embed_dim, num_heads, ff_dim, time_dim)
        self.regressor_mu = nn.Linear(128, 1)
        self.regressor_logvar = nn.Linear(128, 1)

    def forward(self, token_inputs, time_inputs):
        features = self.backbone.forward_features(token_inputs, time_inputs)
        mu = self.regressor_mu(features)
        logvar = self.regressor_logvar(features)
        if self.return_squeezed:
            return mu.squeeze(dim=1), logvar.squeeze(dim=1)
        return mu, logvar


class ProcessTransformerFDS(nn.Module):
    def __init__(self, max_case_length, vocab_size, embed_dim=36, num_heads=4, ff_dim=64, time_dim=3, return_squeezed=True, **config):
        super(ProcessTransformerFDS, self).__init__()
        self.return_squeezed = return_squeezed
        self.backbone = PTBackbone(max_case_length, vocab_size, embed_dim, num_heads, ff_dim, time_dim)
        self.regressor = nn.Linear(128, 1)
        # FDS will be assigned later through factory with proper config
        self.FDS = FDS(feature_dim=128,
                       start_update=config.get('start_update', 0),
                       start_smooth=config.get('start_smooth', 1),
                       kernel=config.get('kernel', 'gaussian'),
                       ks=config.get('ks', 5),
                       sigma=config.get('sigma', 2))

    def forward(self, token_inputs, time_inputs, y, epoch):
        features = self.backbone.forward_features(token_inputs, time_inputs)
        smoothed_features = features
        if self.training and epoch >= self.FDS.start_smooth:
            y_reshaped = y.unsqueeze(1)
            smoothed_features = self.FDS.smooth(smoothed_features, y_reshaped, epoch)
        preds = self.regressor(smoothed_features)
        if self.return_squeezed:
            return {'preds': preds.squeeze(dim=1), 'features': features}
        return {'preds': preds, 'features': features}


class ProcessTransformerFDSMVE(nn.Module):
    def __init__(self, max_case_length, vocab_size, embed_dim=36, num_heads=4, ff_dim=64, time_dim=3, return_squeezed=True, **config):
        super(ProcessTransformerFDSMVE, self).__init__()
        self.return_squeezed = return_squeezed
        self.backbone = PTBackbone(max_case_length, vocab_size, embed_dim, num_heads, ff_dim, time_dim)
        self.regressor_mu = nn.Linear(128, 1)
        self.regressor_logvar = nn.Linear(128, 1)
        self.FDS = FDS(feature_dim=128,
                       start_update=config.get('start_update', 0),
                       start_smooth=config.get('start_smooth', 1),
                       kernel=config.get('kernel', 'gaussian'),
                       ks=config.get('ks', 5),
                       sigma=config.get('sigma', 2))

    def forward(self, token_inputs, time_inputs, y, epoch):
        features = self.backbone.forward_features(token_inputs, time_inputs)
        smoothed_features = features
        if self.training and epoch >= self.FDS.start_smooth:
            y_reshaped = y.unsqueeze(1)
            smoothed_features = self.FDS.smooth(smoothed_features, y_reshaped, epoch)
        preds_mu = self.regressor_mu(smoothed_features)
        preds_logvar = self.regressor_logvar(smoothed_features)
        if self.return_squeezed:
            return {'preds_mu': preds_mu.squeeze(dim=1),
                    'preds_logvar': preds_logvar.squeeze(dim=1),
                    'features': features}
        return {'preds_mu': preds_mu, 'preds_logvar': preds_logvar, 'features': features}