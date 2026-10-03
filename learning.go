package neuraljgo

// LearningObjective defines how a head participates in learning.
//
// The loss is supplied when Learn is called and is evaluated against the head's learning output.
// Learn may validate that the loss is compatible with the head's output configuration.
type LearningObjective struct {
	// Target contains the expected output for the head.
	Target *Tensor

	// Loss calculates the objective value and its gradient.
	Loss Loss

	// Weight scales this head's contribution to the combined learning objective and gradient.
	Weight float64
}

// LearningObjectives defines the learning objective for each selected head.
//
// Only heads present in the map participate in learning. This allows
// individual heads to be trained without requiring targets for all heads.
type LearningObjectives map[HeadID]LearningObjective

// LearningResult contains the results produced by learn operation.
type LearningResult struct {
	// Predictions contains the predictions produced by all network heads
	// before the learning update.
	Predictions map[HeadID]*Tensor

	// HeadLosses contains the loss calculated for each head included
	// in the learning objectives.
	HeadsLoss map[HeadID]float64

	// TotalLoss contains the weighted sum of all head losses.
	TotalLoss float64
}
