import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 
#--------------------RelationNet----------------------------------------------------------------------
# https://github.com/dragen1860/LearningToCompare-Pytorch
class RelationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RelationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(hidden_size * 16 * 16, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input, support):
        input = self.layer1(input)
        input = self.layer2(input)
        input = input.view(input.size(0), -1)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)

        supports = []
        for s in support:
            s = self.layer1(s)
            s = self.layer2(s)
            s = s.view(s.size(0), -1)
            s = F.relu(self.fc1(s))
            s = self.fc2(s)
            supports.append(s)

        supports = torch.stack(supports, dim=1)
        output = input.unsqueeze(1) + supports
        output = torch.sum(output, dim=1)

        return output
#--------------------MatchingNet-----------------------------------------------------------------------
# https://github.com/gitabcworld/MatchingNetworks
class MatchingNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MatchingNet, self).__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, support):
        query = self.layer1(query)
        query = self.layer2(query)
        query = query.view(query.size(0), -1)
        query = self.fc(query)

        supports = []
        for s in support:
            s = self.layer1(s)
            s = self.layer2(s)
            s = s.view(s.size(0), -1)
            s = self.fc(s)
            supports.append(s)

        supports = torch.stack(supports, dim=1)
        output = query.unsqueeze(1) + supports
        output = torch.sum(output, dim=1)

        return output
#--------------------ProtoNet--------------------------------------------------------------------------
# https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/tree/master
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()

    def forward(self, query, support):
        B, _, _, _ = query.size()
        num_class, _, _, _,_ = support.size()
        print(num_class)
        support = support.view(num_class, B, -1)
        query = query.view(B, -1)
        prototypes = support.mean(1)
        distances = self.euclidean_dist(query, prototypes) 
        scores = -distances

        return scores

    def euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
    
#--------------------Cosine classifer------------------------------------------------------------------
# https://github.com/vinuni-vishc/Few-Shot-Cosine-Transformer
class CosineClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, query, support):
        query_features = self.feature_extractor(query)
        support_features = []
        for s in support:
            support_features.append(self.feature_extractor(s))
        support_features = torch.stack(support_features)
        query_features = query_features.view(query_features.size(0), -1)  
        support_features = support_features.view(support_features.size(0), -1)  
        query_features = F.normalize(query_features, dim=1)
        support_features = F.normalize(support_features, dim=1)
        similarities = torch.matmul(query_features, support_features.transpose(0, 1))
        logits = similarities.view(-1, self.num_classes)
        probabilities = F.softmax(logits, dim=1)

        return probabilities
#--------------------CAN-------------------------------------------------------------------------------
# https://github.com/blue-blue272/fewshot-CAN

#--------------------CovaMNet--------------------------------------------------------------------------
# https://github.com/WenbinLee/CovaMNet/tree/master
class CovarianceNet_64(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=10):
		super(CovarianceNet_64, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                       # 3*84*84
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21
		)
		
		self.covariance = CovaBlock()                        # 1*(441*num_classes)

		self.classifier = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=16*16, stride=16*16, bias=use_bias),
		)


	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)
		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			S.append(self.features(input2[i]))

		x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
		x = self.classifier(x)    # get Batch*1*num_classes
		x = x.squeeze(1)          # get Batch*num_classes

		return x

class CovaBlock(nn.Module):
	def __init__(self):
		super(CovaBlock, self).__init__()


	# calculate the covariance matrix 
	def cal_covariance(self, input):
		
		CovaMatrix_list = []
		for i in range(len(input)):
			support_set_sam = input[i]
			B, C, h, w = support_set_sam.size()

			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			mean_support = torch.mean(support_set_sam, 1, True)
			support_set_sam = support_set_sam-mean_support

			covariance_matrix = support_set_sam@torch.transpose(support_set_sam, 0, 1)
			covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
			CovaMatrix_list.append(covariance_matrix)

		return CovaMatrix_list    


	# calculate the similarity  
	def cal_similarity(self, input, CovaMatrix_list):
	
		B, C, h, w = input.size()
		Cova_Sim = []
	
		for i in range(B):
			query_sam = input[i]
			query_sam = query_sam.view(C, -1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)    
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				mea_sim = torch.zeros(1, len(CovaMatrix_list)*h*w).cuda()

			for j in range(len(CovaMatrix_list)):
				temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
				mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

			Cova_Sim.append(mea_sim.unsqueeze(0))

		Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
		return Cova_Sim 


	def forward(self, x1, x2):

		CovaMatrix_list = self.cal_covariance(x2)
		Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

		return Cova_Sim
#--------------------SA CovaMNet-----------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_rate,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate,
                      out_channels=channels,
                      kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # mean on spatial dim
        avg_feat    = torch.mean(x, dim=1, keepdim=True)
        # max on spatial dim
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_rate=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels,
                                                  reduction_rate)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        
        return out

class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm([n_dims, width, height])

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        out = self.norm(out) + x
        return out

class SAModule(nn.Module):
    def __init__(self, w=16, h=16):
        super(SAModule, self).__init__()
        self.mhsa = MHSA(64, w, h)
        self.cbam = CBAM(64)

    def forward(self, x):
        out = self.mhsa(x) + self.cbam(x) + x
        return out

class SA_CovaMNet(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=10):
		super(SA_CovaMNet, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                       # 3*84*84
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21
		)
		
		self.covariance = CovaBlock()                        # 1*(441*num_classes)

		self.classifier = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=16*16, stride=16*16, bias=use_bias),
		)
		self.sa_module = SAModule().cuda()

	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)
		q = self.sa_module(q)
		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			S.append(self.sa_module(self.features(input2[i])))

		x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
		x = self.classifier(x)    # get Batch*1*num_classes
		x = x.squeeze(1)          # get Batch*num_classes

		return x


    
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


#---------------------------QS_Former----------------------------------------------------------
import numpy as np
import pyemd

def emd_distance(tensor1, tensor2):
    array1 = tensor1.cpu().detach().numpy().astype(np.float64)
    array2 = tensor2.cpu().detach().numpy().astype(np.float64)

    flat_array1 = array1.ravel()
    flat_array2 = array2.ravel()

    histogram1, bins = np.histogram(flat_array1, bins=np.arange(flat_array1.min(), flat_array1.max() + 2))
    histogram2, _ = np.histogram(flat_array2, bins=bins)
    histogram1 = histogram1.astype(np.float64)
    histogram2 = histogram2.astype(np.float64)
    cost_matrix = np.abs(np.subtract.outer(flat_array1, flat_array2)).astype(np.float64)

    emd_distance = pyemd.emd(histogram1, histogram2, cost_matrix)

    return emd_distance

class PathFormer(nn.Module):
    def __init__(self, dim=64):
        super(PathFormer, self).__init__()
        self.dim = dim
        self.dot = ScaledDotProductAttention(self.dim)

    def forward(self, q, S):
        q = torch.mean(q, dim =(2,3), keepdim=True)
        q = q.view(q.size(0), q.size(1)) 
        q = q + self.dot(q, q, q)
        out_S = []
        for s in S:
            s = torch.mean(s, dim =(2,3), keepdim=True)
            s = s.view(s.size(0), s.size(1))
            s = s + self.dot(s, s, s)
            out_S.append(s)
        
        out_S = torch.cat(out_S, 0)
        out = []
        for i in range(len(out_S)):
            out.append(emd_distance(q, out_S[i]))
        
        return torch.tensor(out).unsqueeze(0)

class QS_Former(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, dim=64):
        super(QS_Former, self).__init__()
        self.Encoder_Decoder = Encoder_Decoder(dim)
        self.classifier = nn.Conv1d(48, 1, kernel_size=1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
        )

        self.path_former = PathFormer(dim)

    def forward(self, input1, input2):
        q = self.features(input1)
        S = []

        for i in range(len(input2)):
            features_support = self.features(input2[i])
            S.append(features_support)

        # upper branch
        out_u = self.Encoder_Decoder(q, S)
        out_u = torch.cat(out_u, 1)
        # lower branch
        out_l = self.path_former(q, S)
        
        out = out_u + out_l.cuda()
        
        return out

