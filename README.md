# Fire Evacuation ABM

[![Tests](https://github.com/Chadsr/MesaFireEvacuation/actions/workflows/tests.yml/badge.svg)](https://github.com/Chadsr/MesaFireEvacuation/actions/workflows/tests.yml)

The [accompanying paper](./docs/Influence_of_Human_Behaviour_in_the_Evacuation_of_a_Burning_Building.pdf) includes both specification and experimental results. To view the code exactly as it was in the paper, please see [here](https://github.com/Chadsr/MesaFireEvacuation/tree/2597862708a57785b1cb67d4bf527e5896d30a7b).

## Setup
*You will need to install [Poetry](https://python-poetry.org/docs/) using your preferred method, first, then:*

```
cd MesaFireEvacutation/
poetry install
```

## Usage
### Visualised Run

```
poetry run python run.py
```

Runs the model with a visual interface, in which parameters can be changed.

### Batch Run

```
poetry run python run_batch.py <num_iterations> <num_humans>
```

Runs the model with num_iterations of all collaboration factor values with the given num_humans.

## Examples

### Realistic Vision

![vision through smoke](https://github.com/Chadsr/MesaFireEvacuation/blob/master/images/vision.png?raw=true)

As we can see from the figure above, there is an incapacitated agent within the smoke. This agent can not be seen by our healthy agent and will therefore not be helped. We can also observe that the fire is still visible through the smoke, due to its high visibility value.

This functionality is implemented using [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham's_line_algorithm), to create a primitive form of CPU-bound ray-tracing.
