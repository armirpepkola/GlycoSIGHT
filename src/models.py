import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


# HIERARCHICAL FOOD ANALYSIS

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_features=None):
        if skip_features is not None:
            x = torch.cat([x, skip_features], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class HierarchicalFoodAnalysis(nn.Module):

    def __init__(self, num_food_classes=102, pretrained=True):
        super().__init__()
        self.num_classes = num_food_classes

        self.encoder = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            features_only=True,
        )
        encoder_channels = self.encoder.feature_info.channels()

        self.decoder_block1 = DecoderBlock(encoder_channels[4] + encoder_channels[3], 256)
        self.decoder_block2 = DecoderBlock(256 + encoder_channels[2], 128)
        self.decoder_block3 = DecoderBlock(128 + encoder_channels[1], 64)
        self.decoder_block4 = DecoderBlock(64 + encoder_channels[0], 32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(32, self.num_classes, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_channels[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()
        )

        self.volume_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_channels[4], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        s0, s1, s2, s3, s4 = features[0], features[1], features[2], features[3], features[4]

        d1 = self.upsample(s4)
        d1 = self.decoder_block1(d1, s3)

        d2 = self.upsample(d1)
        d2 = self.decoder_block2(d2, s2)

        d3 = self.upsample(d2)
        d3 = self.decoder_block3(d3, s1)

        d4 = self.upsample(d3)
        d4 = self.decoder_block4(d4, s0)

        segmentation_logits = self.segmentation_head(d4)

        pooled_features = self.global_pool(s4)
        classification_probs = self.classification_head(pooled_features)
        volume_estimates = self.volume_head(pooled_features)

        return {
            'segmentation': segmentation_logits,
            'classification': classification_probs,
            'volume': volume_estimates
        }


# NUTRIENT AWARE TRANSFORMER MODULE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NutrientAwareTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.2, num_features=5, output_seq_len=24):
        super().__init__()
        self.d_model = d_model

        self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.positional_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )

        self.output_layer = nn.Linear(d_model, 1)
        self.output_seq_len = output_seq_len

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        embedded_src_list = [self.feature_embeddings[i](src[:, :, i].unsqueeze(-1)) for i in range(src.shape[2])]
        embedded_src = torch.stack(embedded_src_list).sum(dim=0)
        embedded_src = self.positional_encoder(embedded_src * math.sqrt(self.d_model))

        embedded_tgt = self.feature_embeddings[0](tgt)
        embedded_tgt = self.positional_encoder(embedded_tgt * math.sqrt(self.d_model))

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        transformer_out = self.transformer(embedded_src, embedded_tgt, tgt_mask=tgt_mask)
        prediction = self.output_layer(transformer_out)

        return prediction.permute(1, 0, 2)