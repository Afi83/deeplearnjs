import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import {CheckpointLoader, Array1D, Array2D, CostReduction, ENV, Graph, NDArrayMathGPU,
  InCPUMemoryShuffledInputProviderBuilder, NDArray, Scalar, Session, SGDOptimizer, Tensor} from 'deeplearn';

  export interface SampleData {
    images: number[][];
    labels: number[];
  }

  /*
    python dump_checkpoint_vars.py --model_type=tensorflow --output_dir=deeptest\src
    --checkpoint_file=C:\Users\asamiakalantari\AnacondaProjects\TF\DNN\my_model_final.ckpt
  */

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  private varLoader = new CheckpointLoader('../assets/model');
  private sampleData: SampleData;
  private numCorrect: number = 0;
  private math = new NDArrayMathGPU();
  constructor(private http: HttpClient) {

  }
  ngOnInit() {

    this.http.get<SampleData>('../assets/sample_data.json').subscribe(res => {
      this.sampleData = res;
      console.log('Evaluation set #n', this.sampleData.images.length);

      this.varLoader.getAllVariables().then(async vars => {
        for (let i = 0; i < this.sampleData.images.length; i++) {
          const x = Array1D.new(this.sampleData.images[i]);

          const predictedLabel = Math.round(await this.inferModel(x, vars).val());
          const label = this.sampleData.labels[i];
          console.log(`Item ${i}, predicted label ${predictedLabel}. and actual label ${label}`);

          // Aggregate correctness to show accuracy.
          console.log(label === predictedLabel);
          if (label === predictedLabel) {
            this.numCorrect++;
          }
        }

        const accuracy = this.numCorrect * 100 / this.sampleData.images.length;
        console.log('Accuracy is', accuracy);
      });
    });


  }

  inferModel(X: Array1D, vars: {[varName: string]: NDArray}): Scalar {
      // Get NDArray of variables casted with expected dimension.
      const hidden1W = vars['hidden1/kernel'] as Array2D;
      const hidden1B = vars['hidden1/bias'] as Array1D;
      const hidden2W = vars['hidden2/kernel'] as Array2D;
      const hidden2B = vars['hidden2/bias'] as Array1D;
      const softmaxW = vars['outputs/kernel'] as Array2D;
      const softmaxB = vars['outputs/bias'] as Array1D;
      // ...

      // Write your model here.
      const hidden1 = this.math.relu(this.math.add(this.math.vectorTimesMatrix(X, hidden1W), hidden1B)) as Array1D;
      const hidden2 = this.math.relu(this.math.add(this.math.vectorTimesMatrix(hidden1, hidden2W), hidden2B)) as Array1D;

      const logits = this.math.add(this.math.vectorTimesMatrix(hidden2, softmaxW), softmaxB);
      console.log(logits);
      console.log(this.math.argMax(logits));
      return this.math.argMax(logits);
  }
}
