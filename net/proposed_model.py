import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 
import numpy as np

# -------------------Feature Extraction----------------------------------------------------------------

class LKA(nn.Module):
    def __init__(self, dim, kernel_size, dilated_rate=3):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size, padding='same', groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding='same', groups=dim, dilation=dilated_rate)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)

        return u*attn

class my_norm(nn.Module):
    def __init__(self, shape=4096):
        super().__init__()
        self.shape = shape
        self.norm = nn.LayerNorm(shape)
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        x = self.norm(x)
        x = x.view(B,C,H,W)
        return x

class MultiScaleExtractor(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # self.head_pw = nn.Conv2d(dim, dim, 1)
        self.tail_pw = nn.Conv2d(dim, dim, 1)

        self.LKA3 = LKA(dim, kernel_size=3)
        self.LKA5 = LKA(dim, kernel_size=5)
        self.LKA7 = LKA(dim, kernel_size=7)
        self.norm3 = nn.BatchNorm2d(dim)
        self.norm5 = nn.BatchNorm2d(dim)
        self.norm7 = nn.BatchNorm2d(dim)

        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.conv_cn = nn.Conv2d(dim, dim, 3, groups=dim,padding=1)
        self.norm_last = nn.BatchNorm2d(dim)
    def forward(self, x):
        x_copy = x.clone()
        # x = self.head_pw(x)

        x3 = self.LKA3(x) + x
        x3 = self.norm3(x3)
        x5 = self.LKA5(x) + x
        x5 = self.norm5(x5)
        x7 = self.LKA7(x) + x
        x7 = self.norm7(x7)

        x = F.gelu(x3 + x5 + x7)
        x = self.tail_pw(x) + x_copy

        x = self.pointwise(x)
        x = self.conv_cn(x)
        x = F.gelu(self.norm_last(x))
        return x

def Feature_Extractor(dim=64, patch_size=4, depth=2):
    return nn.Sequential(
        nn.Conv2d(1, dim//2, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(1024),
        nn.GELU(),
        nn.Conv2d(dim//2, dim, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(256),
        nn.GELU(),
        *[MultiScaleExtractor(dim=dim) for _ in range(depth)]
    )


#-------------------------------------Mahalanobis block------------------------------------------------------#
class MahalanobisBlock(nn.Module):
    def __init__(self):
        super(MahalanobisBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy() 
                transposed_tensor = np.transpose(local_feature_np, (1, 2, 0))
                reshaped_tensor = np.reshape(transposed_tensor, (h * w, C))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            mean = np.mean(local_feature_np, axis=0)
            local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list, regularization=1e-6):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            # mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w)
            for j in range(len(CovaMatrix_list)):

                covariance_matrix = CovaMatrix_list[j].float().cuda() + regularization * torch.eye(C).cuda()
                # covariance_matrix = CovaMatrix_list[j] + regularization * torch.eye(C)
                inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff.T, inv_covariance_matrix), diff)
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim
    
    
#---------------- Transformer---------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim = 64):
        super(ScaledDotProductAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, q, k, v):
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size, dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attention_weights, v.unsqueeze(2))
        output = output.squeeze(2)
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, q, k, v):
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size,  dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        # output = torch.matmul(attention_weights, v)

        return attention_weights
    
class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.dim = dim
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.FFN = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )

    def forward(self, Support):
      encoded = []
      for index in range(len(Support)):
          s = Support[index]                                  # [B, C, H, W]
          s = self.GAP(s)                                     # [B, C, 1, 1]
          s = s.view(s.size(0), s.size(1))                    # [B, C]
          s = self.ScaledDotProductAttention(s, s, s) + s     # [B, C]
          s = self.norm(s)                                    # [B, C]
          s = self.FFN(s) + s                                 # [B, C]
          s = self.norm(s)                                    # [B, C]
          s = torch.mean(s, dim=0, keepdim=True)
          encoded.append(s)

      return encoded                                          # [Num_class x (B, C)]

class Encoder_Decoder(nn.Module):
    def __init__(self, dim):
        super(Encoder_Decoder, self).__init__()
        self.dim = dim
        self.encoder_out = Encoder(self.dim)
        self.attention = CrossAttention()
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(self.dim ** 2 , 1)
    def forward(self, q, S):
        q = self.GAP(q)                                       # [B, C, 1, 1]
        q = q.view(q.size(0), q.size(1))                      # [B, C]
        q_first = q                                           # [B, C]
        q = self.ScaledDotProductAttention(q, q, q) + q_first # [B, C]
        q = self.norm(q)                                      # [B, C]
        output = []
        encoder_outs = self.encoder_out(S)                    # [Num_class x (B, C)]

        for encoder_out in encoder_outs:
            out = self.attention(q, encoder_out, encoder_out) # [B, C, C]
            out = out.view(out.size(0), -1)                   # [B, C*C]
            out = self.Linear(out)
            output.append(out)


        return output                                         # [Num_class x (B, 1)]

    
#-------------------------------Proposed Ensemble------------------------------------------------
class Ensemble_Net(nn.Module):
    def __init__(self, h=16,w =16,norm_layer=nn.BatchNorm2d, dim=64, alpha1=0.7, alpha2 =0.3):
        self.h = h
        self.w = w
        super(Ensemble_Net, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.covariance = MahalanobisBlock()
        self.Encoder_Decoder = Encoder_Decoder(dim)
        self.features = Feature_Extractor()
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=self.h*self.w, stride=self.h*self.w, bias=use_bias),
        )
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    def forward(self, input1, input2):
        q = self.features(input1)
        S = []
        for i in range(len(input2)):
            features = self.features(input2[i])
            S.append(features)
        # Lower branch
        m_l = self.covariance(q, S)
        m_l = self.classifier(m_l.view(m_l.size(0), 1, -1))
        m_l = m_l.squeeze(1)

        # Upper branch
        m_u = self.Encoder_Decoder(q, S)
        m_u = torch.cat(m_u, 1)
        output = self.alpha1*m_l + self.alpha2*m_u

        return m_l, m_u, output
