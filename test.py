      # 1. Template 피처 먼저 학습 (모든 카테고리 포함)
      template_visu_feats = []
      template_text_feats = []
      template_visu_masks = []
      template_text_masks = []
      template_category_idx = []

      for i in range(bs):
          batch_template_visu_feats = []
          batch_template_text_feats = []
          batch_template_visu_masks = []
          batch_template_text_masks = []

          for j in range(tem_imgs[i].tensors.shape[0]):
              # Visual Encoder for Template
              tem_out, tem_visu_pos = self.visumodel(tem_imgs[i].tensors[j].unsqueeze(0))
              tem_visu_mask, tem_visu_src = tem_out  # Mask와 Visual Feature 분리
              tem_visu_src = self.visu_proj(tem_visu_src)
              batch_template_visu_feats.append(tem_visu_src)
              batch_template_visu_masks.append(tem_visu_mask)

              # Template category index 추가
              tem_category_idx = torch.tensor(self.category_to_idx[tem_cats[i][j]], device=img_data.tensors.device)
              template_category_idx.append(tem_category_idx)

              # Language Encoder for Template
              tem_txt_tensor = tem_txts[i].tensors[j].unsqueeze(0)
              tem_txt_mask = tem_txts[i].mask[j].unsqueeze(0)

              tem_text_fea = self.textmodel(NestedTensor(tem_txt_tensor, tem_txt_mask))
              tem_text_src, tem_text_mask = tem_text_fea.decompose()
              tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)
              batch_template_text_feats.append(tem_text_src)
              batch_template_text_masks.append(tem_text_mask.unsqueeze(0))

          template_visu_feats.append(torch.cat(batch_template_visu_feats, dim=0))
          template_visu_masks.append(torch.cat(batch_template_visu_masks, dim=1))
          template_text_feats.append(torch.cat(batch_template_text_feats, dim=0))
          template_text_masks.append(torch.cat(batch_template_text_masks, dim=1))