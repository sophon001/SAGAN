import logging
from modules import *

class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_input=0.3, finetune=False,
                 dropout_neighbors=0.0,
                 device=torch.device("cpu"),symbol2id=None, symbol2vec=None,e1_rele2=None):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False
       
        self.aggregation = Aggregation(dim = 100, dropout=dropout_input,e1_rele2=e1_rele2)
   
    
    def entity_encoder(self, connections_left, connections_right, head_left, head_right):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        relations_left = connections_left[:, :, 0].squeeze(-1)
        entities_left = connections_left[:, :, 1].squeeze(-1)
        rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]
        ent_embeds_left = self.dropout(self.symbol_emb(entities_left))

        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        mask_left = torch.eq(relations_left, pad_matrix_left).squeeze(-1)  # [b, max]

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (batch, 200, embed_dim)
        ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (batch, 200, embed_dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_right = torch.eq(relations_right, pad_matrix_right).squeeze(-1)  # [b, max]

        left = [head_left, rel_embeds_left, ent_embeds_left]
        right = [head_right, rel_embeds_right, ent_embeds_right]
       
        entity_left, entity_right = self.aggregation(left, right,connections_left,connections_right)

        return entity_left, entity_right


    def forward(self, entity, entity_meta=None): 
        '''
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         '''
        if entity_meta is not None:
            entity = self.symbol_emb(entity)
            entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
            entity_left, entity_right = self.entity_encoder(entity_left_connections,
                                                                          entity_right_connections,
                                                                          entity_left, entity_right)
        else:
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
        return entity_left, entity_right

class Sagan_encoder(nn.Module):
    def __init__(self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(Sagan_encoder, self).__init__()
        self.Encoder = TransformerEncoder(model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3,
                                                  with_pos=True)

    def forward(self, left, right):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return: [batch, dim]
        """

        relation = self.Encoder(left, right)
        return relation

class Matcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_layers=0.1, dropout_input=0.3,
                 dropout_neighbors=0.0,
                 finetune=False, num_transformer_layers=6, num_transformer_heads=4,
                 device=torch.device("cpu"),symbol2id=None, symbol2vec=None,e1_rele2=None,
                 ):
        super(Matcher, self).__init__()
        self.EntityEncoder = EntityEncoder(embed_dim, num_symbols,
                                           use_pretrain=use_pretrain,
                                           embed=embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors,
                                           finetune=finetune, device=device,symbol2id=symbol2id, symbol2vec=symbol2vec,e1_rele2=e1_rele2)
        self.entity_pair_encoder = Sagan_encoder(emb_dim=embed_dim,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.Prototype = SemanticPrototypeNet(embed_dim * num_transformer_heads) 


    def forward(self, support, query, false=None, isEval=False, support_meta=None, query_meta=None, false_meta=None):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """
        if not isEval:
            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)
            false_r = self.EntityEncoder(false, false_meta)

            support_r = self.entity_pair_encoder(support_r[0], support_r[1])
            query_r = self.entity_pair_encoder(query_r[0], query_r[1])
            false_r = self.entity_pair_encoder(false_r[0], false_r[1])

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:

            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)

            support_r = self.entity_pair_encoder(support_r[0], support_r[1])
            query_r = self.entity_pair_encoder(query_r[0], query_r[1])

   
            center_q = self.Prototype(support_r, query_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
        return positive_score, negative_score

