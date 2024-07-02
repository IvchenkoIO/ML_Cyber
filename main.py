import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Activation, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from jinja2 import Template
import matplotlib.pyplot as plt
import os

tfd = tfp.distributions

class MDNRNNModel:
    def __init__(self, input_shape, output_shape, num_mixtures, lstm_units, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_mixtures = num_mixtures
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.get_mixture_loss())

    def get_mixture_loss(self):
        def mixture_loss(y_true, y_pred):
            y_true = K.expand_dims(y_true, axis=-1)
            pi = y_pred[:, :self.num_mixtures]
            mu = y_pred[:, self.num_mixtures:2*self.num_mixtures]
            sigma = y_pred[:, 2*self.num_mixtures:]
            
            pi = K.softmax(pi)
            sigma = K.exp(sigma)
            
            dist = tfd.Normal(loc=mu, scale=sigma)
            prob = dist.prob(y_true)
            prob = K.sum(pi * prob, axis=-1)
            loss = -K.log(prob + 1e-8)
            return K.mean(loss)
        return mixture_loss

    def build_model(self):
        inputs = Input(shape=self.input_shape, name='Input_Layer')
        lstm_1 = LSTM(self.lstm_units, return_sequences=True, name='LSTM_Layer_1')(inputs)
        lstm_2 = LSTM(self.lstm_units, return_sequences=True, name='LSTM_Layer_2')(lstm_1)
        lstm_3 = LSTM(self.lstm_units, name='LSTM_Layer_3')(lstm_2)
        
        mdn_output = Dense(self.num_mixtures * 3, name='MDN_Output')(lstm_3)
        
        pi = Lambda(lambda x: x[:, :self.num_mixtures], name='Pi_Lambda')(mdn_output)
        mu = Lambda(lambda x: x[:, self.num_mixtures:2*self.num_mixtures], name='Mu_Lambda')(mdn_output)
        sigma = Lambda(lambda x: x[:, 2*self.num_mixtures:], name='Sigma_Lambda')(mdn_output)
        
        pi = Activation('softmax', name='Pi_Activation')(pi)
        sigma = Activation('exponential', name='Sigma_Activation')(sigma)
        
        output = concatenate([pi, mu, sigma], name='Output_Concatenate')
        return Model(inputs=inputs, outputs=output, name='MDN_RNN_Model')

    def train(self, x_train, y_train, epochs=50, batch_size=64):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def print_data_shapes(self, x_data, y_data, data_name=""):
        print(f"{data_name} Input Data Shape: {x_data.shape}\n{x_data}")
        if y_data is not None:
            print(f"{data_name} Output Data Shape: {y_data.shape}\n{y_data}")
        else:
            print(f"{data_name} Output Data is None")

    def analyze_predictions(self, predictions, y_test):
        analyses = []
        for i in range(predictions.shape[0]):
            pi = predictions[i, :self.num_mixtures]
            mu = predictions[i, self.num_mixtures:2*self.num_mixtures]
            sigma = predictions[i, 2*self.num_mixtures:]
            
            predicted_value = np.sum(pi * mu)
            actual_value = y_test[i, 0]
            error = actual_value - predicted_value
            
            analysis = {
                'index': i,
                'predicted_value': predicted_value,
                'actual_value': actual_value,
                'error': error,
                'abs_error': abs(error),
                'pi': pi.tolist(),
                'mu': mu.tolist(),
                'sigma': sigma.tolist()
            }
            analyses.append(analysis)
        
        mean_abs_error = np.mean([a['abs_error'] for a in analyses])
        max_abs_error = np.max([a['abs_error'] for a in analyses])
        min_abs_error = np.min([a['abs_error'] for a in analyses])
        
        summary = {
            'mean_abs_error': mean_abs_error,
            'max_abs_error': max_abs_error,
            'min_abs_error': min_abs_error
        }
        
        return analyses, summary

    def save_predictions_to_html(self, x_test, y_test, predictions, analyses, summary, file_name='predictions.html'):
        if not os.path.exists('plots'):
            os.makedirs('plots')

        for i in range(predictions.shape[0]):
            plt.figure(figsize=(10, 6))
            pi = predictions[i, :self.num_mixtures]
            mu = predictions[i, self.num_mixtures:2*self.num_mixtures]
            sigma = predictions[i, 2*self.num_mixtures:]
            
            for j in range(self.num_mixtures):
                plt.plot(mu[j], pi[j], 'o')
                plt.errorbar(mu[j], pi[j], xerr=sigma[j])
            plt.title(f'Mixture Components for Prediction {i}')
            plt.xlabel('Mu')
            plt.ylabel('Pi')
            plt.savefig(f'plots/prediction_{i}.png')
            plt.close()

        template = Template("""
            <html>
            <head>
                <title>MDN-RNN Predictions - Nurlybek Maghzan</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }
                    table, th, td {
                        border: 1px solid #ddd;
                    }
                    th, td {
                        padding: 15px;
                        text-align: left;
                    }
                    tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                    th {
                        background-color: #2c3e50;
                        color: white;
                    }
                    summary {
                        cursor: pointer;
                        font-weight: bold;
                        background-color: #ecf0f1;
                        padding: 10px;
                        border: 1px solid #bdc3c7;
                        border-radius: 5px;
                    }
                    details[open] summary {
                        background-color: #bdc3c7;
                    }
                    details > div {
                        padding: 10px;
                        border: 1px solid #bdc3c7;
                        border-radius: 5px;
                        margin-bottom: 10px;
                    }
                    .section {
                        margin-bottom: 30px;
                    }
                    .section p {
                        margin: 10px 0;
                    }
                </style>
            </head>
            <body>
                <h1>MDN-RNN Predictions - <i>Nurlybek Maghzan</i> </h1>
            
                <p>Этот отчет содержит анализ предсказаний модели MDN-RNN. В нем представлены входные данные, предсказания модели, выходные данные и сводка анализа.</p>

                <h2>Введение</h2>
                <p>MDN-RNN (Mixture Density Network - Recurrent Neural Network) - это модель, которая сочетает в себе рекуррентные нейронные сети (RNN) с сетями смеси плотностей (MDN). Такая комбинация позволяет прогнозировать временные ряды и неопределенные данные, что особенно полезно в задачах кибербезопасности для оценки вероятности различных инцидентов и разработки стратегий защиты.</p>
                <p>В данной работе использовалась модель MDN-RNN для прогнозирования инцидентов кибербезопасности в системе управления железнодорожными перевозками. Модель анализирует временные и числовые характеристики данных и подбирает оптимальные стратегии защиты информации.</p>

                <h2>Цель</h2>
                <p>Целью данного проекта было разработать и обучить модель MDN-RNN для прогнозирования инцидентов кибербезопасности и оценки оптимальных стратегий защиты информации. Входные данные представляют собой временные ряды характеристик системы, а выходные данные - вероятности, средние значения и стандартные отклонения для различных сценариев инцидентов.</p>

                <details open class="section">
                    <summary>Input Data</summary>
                    <div>
                        <p>Входные данные представляют собой последовательности характеристик, которые используются моделью для предсказания.</p>
                        <table>
                            <tr>
                                <th>Index</th>
                                <th>Values</th>
                            </tr>
                            {% for i in range(x_test.shape[0]) %}
                                <tr>
                                    <td>{{ i }}</td>
                                    <td>{{ x_test[i] }}</td>
                                </tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>

                <details class="section">
                    <summary>Predictions</summary>
                    <div>
                        <p>В этом разделе представлены предсказания модели, включая вероятности компонентов (Pi), средние значения (Mu) и стандартные отклонения (Sigma).</p>
                        {% for i in range(predictions.shape[0]) %}
                            <details class="section">
                                <summary>Prediction {{ i }}</summary>
                                <div>
                                    <img src="plots/prediction_{{ i }}.png" alt="Prediction {{ i }}">
                                    <table>
                                        <tr>
                                            <th>Component</th>
                                            <th>Pi</th>
                                            <th>Mu</th>
                                            <th>Sigma</th>
                                        </tr>
                                        {% for j in range(num_mixtures) %}
                                            <tr>
                                                <td>{{ j }}</td>
                                                <td>{{ predictions[i, j] }}</td>
                                                <td>{{ predictions[i, num_mixtures + j] }}</td>
                                                <td>{{ predictions[i, 2 * num_mixtures + j] }}</td>
                                            </tr>
                                        {% endfor %}
                                    </table>
                                    <p><strong>Predicted Value:</strong> {{ analyses[i].predicted_value }}</p>
                                    <p><strong>Actual Value:</strong> {{ analyses[i].actual_value }}</p>
                                    <p><strong>Error:</strong> {{ analyses[i].error }}</p>
                                </div>
                            </details>
                        {% endfor %}
                    </div>
                </details>

                <details class="section">
                    <summary>Output Data</summary>
                    <div>
                        <p>Выходные данные представляют собой фактические значения, которые модель пыталась предсказать.</p>
                        <table>
                            <tr>
                                <th>Index</th>
                                <th>Values</th>
                            </tr>
                            {% for i in range(y_test.shape[0]) %}
                                <tr>
                                    <td>{{ i }}</td>
                                    <td>{{ y_test[i] }}</td>
                                </tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>

                <details class="section">
                    <summary>Summary</summary>
                    <div>
                        <p>Этот раздел содержит сводку анализа предсказаний, включая среднюю абсолютную ошибку (MAE), максимальную и минимальную абсолютные ошибки.</p>
                        <p><strong>Mean Absolute Error:</strong> {{ summary.mean_abs_error }}</p>
                        <p><strong>Max Absolute Error:</strong> {{ summary.max_abs_error }}</p>
                        <p><strong>Min Absolute Error:</strong> {{ summary.min_abs_error }}</p>
                    </div>
                </details>

                <details class="section">
                    <summary>Conclusions</summary>
                    <div>
                        <p><strong>Выводы:</strong></p>
                        <p>Анализ предсказаний модели MDN-RNN показал следующие результаты:</p>
                        <ul>
                            <li><strong>Средняя абсолютная ошибка (MAE):</strong> {{ summary.mean_abs_error }} - это означает, что в среднем модель ошибается на это значение при предсказании значений.</li>
                            <li><strong>Максимальная абсолютная ошибка:</strong> {{ summary.max_abs_error }} - наибольшая ошибка предсказания, показывающая крайние случаи, когда модель значительно отклонилась от фактического значения.</li>
                            <li><strong>Минимальная абсолютная ошибка:</strong> {{ summary.min_abs_error }} - наименьшая ошибка предсказания, показывающая случаи, когда модель была наиболее точна.</li>
                        </ul>
                        <p>Эти метрики помогают оценить качество модели и понять, насколько хорошо она справляется с предсказанием данных. Средняя абсолютная ошибка (MAE) указывает на общую точность модели, а максимальная и минимальная ошибки помогают выявить случаи, когда модель была наиболее и наименее точной соответственно. На основе этих данных можно сделать вывод о том, насколько надежна модель и где возможны улучшения.</p>
                        <p>Прогнозирование инцидентов кибербезопасности с помощью MDN-RNN позволяет заранее оценивать вероятности различных угроз и разрабатывать стратегии для их предотвращения. Это помогает повысить безопасность информации и обеспечить устойчивость системы к потенциальным атакам.</p>
                    </div>
                </details>
            </body>
            </html>


        """)
        html_content = template.render(predictions=predictions, num_mixtures=self.num_mixtures, x_test=x_test, y_test=y_test, analyses=analyses, summary=summary)
        with open(file_name, 'w') as f:
            f.write(html_content)

# Parameters
input_shape = (None, 15)
output_shape = 1
num_mixtures = 10
lstm_units = 100

# Create the model
mdn_rnn = MDNRNNModel(input_shape, output_shape, num_mixtures, lstm_units)

# Generate synthetic data
x_train = np.random.rand(1000, 20, 15)
y_train = np.random.rand(1000, 1)
x_test = np.random.rand(100, 20, 15)
y_test = np.random.rand(100, 1)

# Print data shapes
mdn_rnn.print_data_shapes(x_train, y_train, "Training")
mdn_rnn.print_data_shapes(x_test, y_test, "Test")

# Train the model
mdn_rnn.train(x_train, y_train)

# Make predictions
predictions = mdn_rnn.predict(x_test)

# Analyze predictions
analyses, summary = mdn_rnn.analyze_predictions(predictions, y_test)

# Print predictions
print(f"Predictions Shape: {predictions.shape}\n{predictions}")

# Save predictions to HTML
mdn_rnn.save_predictions_to_html(x_test, y_test, predictions, analyses, summary)
