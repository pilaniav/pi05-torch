from pi05.plot_log import parse_log


def test_parse_log_reads_train_and_val_lines(tmp_path):
    """
    Train and val lines are parsed past the logging prefix; a repeated step keeps the
    later line (resume).
    """
    log = tmp_path / "train.log"
    log.write_text(
        "2026-11-15 16:00:00 INFO pi05.train: step 100  loss 0.1900  grad_norm 0.500"
        "  lr 2.50e-06  0.66s/step  mem 65.4GB\n"
        "2026-11-15 16:01:00 INFO pi05.train: step 100  val_loss 0.0580\n"
        "2026-11-15 16:02:00 INFO pi05.train: Resuming from step 0\n"
        "2026-11-15 16:03:00 INFO pi05.train: step 100  loss 0.1800  grad_norm 0.400"
        "  lr 2.50e-06  0.66s/step  mem 65.4GB\n"
    )

    train, val = parse_log(log)

    assert train == {100: (0.18, 0.4, 2.5e-06)}
    assert val == {100: 0.058}