import json, os, gc, torch, logging, re
from bert_dataset import BertDataset
from metric import TokenMetric
from model.SyntaxBert import SyntaxBertForTokenClassification
from model.bert import SyntaxBertConfig
from utils import set_random_seed, DataProcessor, FeaturizedDataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup


class TextCorrector(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.processor = DataProcessor(config.data_path)
        config_json_file = os.path.join(config.save_path, "SyntaxBertConfig.json")
        self.config = config
        bert_config = SyntaxBertConfig.from_pretrained(config_json_file)
        bert_config.num_labels_classes = self.processor.num_labels_classes
        bert_config.num_detect_classes = self.processor.num_detect_classes
        self.model = SyntaxBertForTokenClassification.from_pretrained(config.save_path, config=
        bert_config, label_map=self.processor.label_map)
        self.TC_evaluation_metric = TokenMetric(labels=self.processor.get_id2label())
        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(config.save_path, "eval_result_log.txt"),
                            level=logging.CRITICAL)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.CRITICAL)

    def get_dataloader(self, prefix="train"):
        examples = self.processor.get_examples(prefix)
        dataset = BertDataset(self.config.data_path, max_seq_length=self.config.max_seqlength, prefix=prefix)
        dataset.examples = examples
        self.result_logger.critical(f"***** Running {prefix} / Num Examples {len(dataset)} *****")
        if prefix == "train":
            batch_size = self.config.train_batch_size
            data_generator = torch.Generator()
            data_generator.manual_seed(self.config.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.config.eval_batch_size
            data_sampler = SequentialSampler(dataset)
        # sampler option is mutually exclusive with shuffle
        dataloader = FeaturizedDataLoader(dataset, gpus=self.config.gpus, prefix=prefix, batch_size=batch_size, sampler=data_sampler)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train")  # validation测试程序 train

    def training_step(self, batch, batch_idx):
        output_dict = self.model.forward(**batch)
        tf_board_logs = {"train_loss": output_dict['loss'], "lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        return {"loss": output_dict['loss'], 'log': tf_board_logs}

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def validation_step(self, batch, batch_idx):
        output_dict = self.model.forward(**batch)
        self.TC_evaluation_metric.update(output_dict['predict'], batch['labels'], training=False)
        tf_board_logs = {"val_loss": output_dict['loss']}
        return {"val_loss": output_dict['loss'], 'log': tf_board_logs}

    def validation_epoch_end(self, outputs):
        all_metrics = self.TC_evaluation_metric.get_stats(training=False)
        self.result_logger.critical(f"EVAL INFO -> current_epoch: {self.trainer.current_epoch}| current_global_step: "
                                    f"{self.trainer.global_step}")
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.result_logger.critical("EVAL INFO -> val_loss: {:.3f}| val_f: {:.3f}| Precision: {:.3f}| Recall: {:.3f}"
                                    "| val_acc: {:.3f}".format(avg_loss, all_metrics['f'], all_metrics['precision'],
                                                              all_metrics['recall'], all_metrics['accuracy']))
        return {"val_loss": avg_loss, "val_f": torch.tensor(all_metrics['f'], dtype=torch.float32), 'val_acc': torch.tensor(all_metrics['accuracy'], dtype=torch.float32)}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        del no_decay
        if self.config.optimizer == "adamw":  # according to RoBERTa paper
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=self.config.lr,
                              eps=self.config.adam_epsilon)
        elif self.config.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=self.config.adam_epsilon,
                                          weight_decay=self.config.weight_decay)
        else:
            raise ValueError("Please import the Optimizer first. ")
        del optimizer_grouped_parameters
        torch.cuda.empty_cache()
        t_total = (len(self.train_dataloader()) // self.config.accumulate_grad_batches) * self.config.max_epochs
        warmup_steps = int(self.config.warmup_proportion * t_total)
        if self.config.no_lr_scheduler:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            out_dict = self.model.forward(**batch)
        predicts = torch.split(out_dict['predict'], out_dict['lens'])
        self.TC_evaluation_metric.save_predictions_to_file(predicts, out_dict['max_error_probability'],
                                              batch['input_ids'])


# 训练的参数设定表
class configurations:
    def __init__(self):
        self.data_path = './data_vocab'
        self.seed = 2020
        self.save_path = './out'
        self.max_seqlength = 150  # 250
        self.lr = 2e-5
        self.weight_decay = 0.002
        self.warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for
        self.accumulate_grad_batches = 2
        self.adam_epsilon = 1e-8
        self.optimizer = "torch.adam"
        self.no_lr_scheduler = False
        self.max_epochs = 1
        self.save_topk = 5
        self.val_check_interval = 4000  # 0.25:每训练单个epoch的 25% 调用校验函数一次 2000
        self.gpus = '1'  # 0
        self.train_batch_size = 2  # 10
        self.eval_batch_size = 2  # 15
        self.pretrained = False  # True
        self.predict = False  # True


def main():
    config = configurations()
    print('config:\n', vars(config))
    gc.collect()
    torch.cuda.empty_cache()
    set_random_seed(config.seed)
    # create save path if doesn't exit
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    model = TextCorrector(config)
    logger = TensorBoardLogger(save_dir=config.save_path, name='log')
    if config.predict:
        trainer = Trainer(logger=logger, gpus=config.gpus, deterministic=True)
        # load一个模型，包括它的weights、biases和超参数
        best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(config.save_path)
        model.result_logger.critical("=&" * 20)
        model.result_logger.critical(f"Best f1 on DEV is {best_f1_on_dev}")
        model.result_logger.critical(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
        checkpoint = torch.load(path_to_best_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.test(model)
        model.result_logger.critical("=&" * 20)
    else:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(config.save_path, 'checkpoint', 'TextCorrector-{epoch}'),
            save_top_k=config.save_topk, monitor="val_f", mode="max", verbose=True, save_weights_only=True, period=-1)
        # save args
        with open(os.path.join(config.save_path, 'checkpoint', "config.json"), 'w') as f:
            config_dict = config.__dict__
            json.dump(config_dict, f, indent=4)  # indent用于json文件中的缩进（每个属性都缩进了4个空格）
        trainer = Trainer(logger=logger, checkpoint_callback=checkpoint_callback, gpus=config.gpus,
                          max_epochs=config.max_epochs, val_check_interval=config.val_check_interval,
                          deterministic=True)
        if config.pretrained:
            _, path_to_checkpoint = find_best_checkpoint_on_dev(config.save_path)
            checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            trainer.global_step = checkpoint['global_step']

        trainer.fit(model)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()
    F1_PATTERN = re.compile(r"val_f reached \d+\.\d* \(best")
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")
        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt
    return best_f1_on_dev, best_checkpoint_on_dev


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()