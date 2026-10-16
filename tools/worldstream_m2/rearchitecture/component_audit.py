"""Equal-data batch-shape audit: isolate norm, QKV rounding, attention and MLP."""
import torch
import torch.nn.functional as F
from worldstream.m2.packing import partition,unpartition
from .common import ROOT,write,error_location


@torch.no_grad()
def run(model,images):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    x=model.patch_embed(images)
    embedding_micro=torch.cat([model.patch_embed(c[None]) for c in images])
    embedding_error=error_location(embedding_micro,x)
    x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
    rows=[]
    for index in (0,1):
        block=model.blocks[index];size=block.window_size
        raw,pad=partition(x,size);normal,_=partition(block.norm1(x),size)
        valid,_=partition(torch.ones_like(x[...,:1]),size)
        attn_full=block.attn(normal)
        residual=x+unpartition(attn_full,size,pad,x.shape[1:3])
        norm2=block.norm2(residual)
        mlp_full,_=partition(block.mlp(norm2),size)
        norm2_windows,_=partition(norm2,size)
        for count in (1,6,36):
            ids=torch.arange(count,device=x.device)
            normal_selected=block.norm1(raw.index_select(0,ids))*valid.index_select(0,ids)
            metrics=dict(norm1=error_location(normal_selected,normal.index_select(0,ids)))
            for name,bias in [('q',block.attn.q_bias),('k',None),('v',block.attn.v_bias)]:
                weight=getattr(block.attn,name+'_proj').weight
                a=F.linear(normal.reshape(54,256,-1),weight,bias)
                b=F.linear(normal_selected.reshape(count,256,-1),weight,bias)
                metrics[name+'_projection_fp32']=error_location(b,a.index_select(0,ids))
                def rope_half(t):
                    t=t.reshape(len(t),256,block.attn.num_heads,-1).permute(0,2,1,3)
                    return (block.attn.rope(t) if name!='v' else t).half()
                aa=rope_half(a).index_select(0,ids);bb=rope_half(b)
                metrics[name+'_flash_input_fp16_changed_elements']=int((aa!=bb).sum())
            selected_attn=block.attn(normal_selected)
            metrics['attention_output']=error_location(selected_attn,attn_full.index_select(0,ids))
            mlp_selected=block.mlp(norm2_windows.index_select(0,ids))
            # Padding outputs are intentionally excluded: original MLP has no padding tokens.
            mask=valid.index_select(0,ids).bool().expand_as(mlp_selected)
            metrics['mlp_same_input_valid_only']=error_location(mlp_selected[mask],mlp_full.index_select(0,ids)[mask])
            rows.append(dict(block=index,selected_windows=count,metrics=metrics,attention_scale=block.attn.scale,
                flash_softmax_scale=getattr(getattr(block.attn,'inner_attn',None),'softmax_scale',None),
                norm_eps=[block.norm1.eps,block.norm2.eps],padding='norm-before-pad, invalid MLP outputs excluded'))
        x=block(x)
    write(ROOT/'component_numerical_audit.json',dict(rows=rows,patch_embedding_microbatch=embedding_error,
        matmul_tf32_allowed=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32_allowed=torch.backends.cudnn.allow_tf32,
        scope='same token values, no historical reuse; isolates batch/kernel implementation differences'))
