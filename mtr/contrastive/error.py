import torch
import os
import pandas as pd

# Compute recall@k
def recall():
    caption_embs = torch.load(os.path.join("../ecals_result", "caption_embs.pt"))
    
    track_ids = list(caption_embs.keys())

    target_items = {caption_embs[k]['text']:k for k in track_ids}

    audio_embs = torch.stack([caption_embs[k]['z_audio'] for k in track_ids])
    text_embs = torch.cat([caption_embs[k]['z_text'] for k in track_ids], dim=0)
    audio_embs = torch.nn.functional.normalize(audio_embs, dim=1)
    text_embs = torch.nn.functional.normalize(text_embs, dim=1)

    logits = text_embs @ audio_embs.T # text to audio
    df_pred = pd.DataFrame(logits.numpy(), index=target_items.keys(), columns=target_items.values())
    pred_items = {}
    for idx in range(len(df_pred)):
        item = df_pred.iloc[idx]
        pred_items[item.name] = list(item.sort_values(ascending=False).index)
        print(pred_items[item.name][:5])
    


if __name__ == "__main__":
    recall()