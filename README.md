
# How to run this code
## 1. Download the dataset from Google  Drive

[NELL-One and Wiki-One Datasets](https://drive.google.com/file/d/1M5JyJGKR71VYE9U4MAk4tvQesiXdZxGO/view?usp=sharing)


## 2. File structure
```bash
.
|-- NELL
|-- README.md
|-- args.py
|-- data_loader.py
|-- matcher.py
|-- modules.py
|-- trainer.py
`-- wiki
```

## 3. Execute the following script
#### For NELL-One dataset

```bash
python trainer.py --weight_decay 0.0 --prefix nell.5shot
```

#### For Wiki-One dataset

```bash
python trainer.py --dataset wiki --embed_dim 50 --num_transformer_layers 4 --num_transformer_heads 8 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot
```

