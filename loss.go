package neuraljgo

import (
	"fmt"
)

type Loss interface {
	Forward(predicted *Tensor, target *Tensor) (float64, error)
	Gradient(predicted *Tensor, target *Tensor) (*Tensor, error)
}

func requireLossInput(predicted *Tensor, target *Tensor) error {
	if predicted == nil {
		return fmt.Errorf("predicted/logits tensor is nil")
	}
	if err := predicted.RequireRank(1, "predicted/logits tensor"); err != nil {
		return err
	}
	if target == nil {
		return fmt.Errorf("target tensor is nil")
	}
	if err := target.RequireRank(1, "target tensor"); err != nil {
		return err
	}
	if predicted.Size() != target.Size() {
		return fmt.Errorf("predicted/logits and target tensors size mismatch: %d != %d", predicted.Size(), target.Size())
	}

	if predicted.DataSize() != target.DataSize() {
		return fmt.Errorf("fatal internal error: predicted/logits and target tensors data size mismatch: %d != %d", predicted.DataSize(), target.DataSize())
	}

	if predicted.Size() != predicted.DataSize() {
		return fmt.Errorf("fatal internal error: predicted tensor size and data size mismatch: %d != %d", predicted.Size(), predicted.DataSize())
	}

	return nil
}
