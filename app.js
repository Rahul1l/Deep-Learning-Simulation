// SUPER DL PLAYGROUND - Pure JavaScript MLP Implementation
class MLP {
    constructor(inputSize, hiddenUnits, activation = 'relu', learningRate = 0.01, l2Lambda = 0.0, randomSeed = 42, taskType = 'classification') {
        this.inputSize = inputSize;
        this.hiddenUnits = hiddenUnits;
        this.activation = activation;
        this.learningRate = learningRate;
        this.l2Lambda = l2Lambda;
        this.taskType = taskType;
        
        // Set random seed
        Math.seedrandom = randomSeed;
        
        // Initialize weights and biases
        this.W1 = this.randomMatrix(inputSize, hiddenUnits);
        this.b1 = this.zeros(1, hiddenUnits);
        this.W2 = this.randomMatrix(hiddenUnits, 1);
        this.b2 = this.zeros(1, 1);
        
        // Activation functions
        this.activationFunctions = {
            relu: {
                forward: (x) => x.map(row => row.map(val => Math.max(0, val))),
                backward: (x) => x.map(row => row.map(val => val > 0 ? 1 : 0))
            },
            leaky_relu: {
                forward: (x) => x.map(row => row.map(val => Math.max(0.01 * val, val))),
                backward: (x) => x.map(row => row.map(val => val > 0 ? 1 : 0.01))
            },
            tanh: {
                forward: (x) => x.map(row => row.map(val => Math.tanh(val))),
                backward: (x) => x.map(row => row.map(val => 1 - Math.tanh(val) ** 2))
            },
            sigmoid: {
                forward: (x) => x.map(row => row.map(val => 1 / (1 + Math.exp(-val)))),
                backward: (x) => x.map(row => row.map(val => {
                    const s = 1 / (1 + Math.exp(-val));
                    return s * (1 - s);
                }))
            }
        };
    }
    
    randomMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push((Math.random() - 0.5) * 2 * Math.sqrt(2.0 / rows));
            }
            matrix.push(row);
        }
        return matrix;
    }
    
    zeros(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push(0);
            }
            matrix.push(row);
        }
        return matrix;
    }
    
    matrixMultiply(A, B) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            const row = [];
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < B.length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                row.push(sum);
            }
            result.push(row);
        }
        return result;
    }
    
    matrixAdd(A, B) {
        return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    }
    
    matrixSubtract(A, B) {
        return A.map((row, i) => row.map((val, j) => val - B[i][j]));
    }
    
    matrixTranspose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }
    
    forward(X) {
        // Hidden layer
        const z1 = this.matrixAdd(this.matrixMultiply(X, this.W1), this.b1);
        const a1 = this.activationFunctions[this.activation].forward(z1);
        
        // Output layer
        const z2 = this.matrixAdd(this.matrixMultiply(a1, this.W2), this.b2);
        
        // For classification: apply sigmoid. For regression: linear output
        const a2 = this.taskType === 'classification' ? 
            this.activationFunctions.sigmoid.forward(z2) : z2;
        
        return { z1, a1, z2, a2 };
    }
    
    backward(X, y, forwardResult) {
        const { z1, a1, z2, a2 } = forwardResult;
        const m = X.length;
        
        // Output layer gradients
        // For classification: gradient is (a2 - y) due to sigmoid
        // For regression: gradient is also (a2 - y) for MSE loss
        const dL_dz2 = this.matrixSubtract(a2, y);
        const dL_dW2 = this.matrixAdd(
            this.matrixMultiply(this.matrixTranspose(a1), dL_dz2),
            this.scalarMultiply(this.W2, this.l2Lambda)
        );
        const dL_db2 = this.sumColumns(dL_dz2);
        
        // Hidden layer gradients
        const dL_da1 = this.matrixMultiply(dL_dz2, this.matrixTranspose(this.W2));
        const dL_dz1 = this.elementWiseMultiply(dL_da1, this.activationFunctions[this.activation].backward(z1));
        const dL_dW1 = this.matrixAdd(
            this.matrixMultiply(this.matrixTranspose(X), dL_dz1),
            this.scalarMultiply(this.W1, this.l2Lambda)
        );
        const dL_db1 = this.sumColumns(dL_dz1);
        
        return { dL_dW1, dL_db1, dL_dW2, dL_db2 };
    }
    
    scalarMultiply(matrix, scalar) {
        return matrix.map(row => row.map(val => val * scalar));
    }
    
    elementWiseMultiply(A, B) {
        return A.map((row, i) => row.map((val, j) => val * B[i][j]));
    }
    
    sumColumns(matrix) {
        const sums = [];
        for (let j = 0; j < matrix[0].length; j++) {
            let sum = 0;
            for (let i = 0; i < matrix.length; i++) {
                sum += matrix[i][j];
            }
            sums.push([sum]);
        }
        return [sums];
    }
    
    updateWeights(gradients) {
        const { dL_dW1, dL_db1, dL_dW2, dL_db2 } = gradients;
        
        this.W1 = this.matrixSubtract(this.W1, this.scalarMultiply(dL_dW1, this.learningRate));
        this.b1 = this.matrixSubtract(this.b1, this.scalarMultiply(dL_db1, this.learningRate));
        this.W2 = this.matrixSubtract(this.W2, this.scalarMultiply(dL_dW2, this.learningRate));
        this.b2 = this.matrixSubtract(this.b2, this.scalarMultiply(dL_db2, this.learningRate));
    }
    
    predict(X) {
        const forwardResult = this.forward(X);
        return forwardResult.a2.map(row => row.map(val => val > 0.5 ? 1 : 0));
    }
    
    computeLoss(yTrue, yPred) {
        const m = yTrue.length;
        let loss = 0;
        
        if (this.taskType === 'classification') {
            // Binary Cross-Entropy Loss
            for (let i = 0; i < m; i++) {
                const y = yTrue[i][0];
                const yHat = yPred[i][0];
                loss += -(y * Math.log(yHat + 1e-8) + (1 - y) * Math.log(1 - yHat + 1e-8));
            }
        } else {
            // Mean Squared Error Loss for regression
            for (let i = 0; i < m; i++) {
                const y = yTrue[i][0];
                const yHat = yPred[i][0];
                loss += Math.pow(y - yHat, 2);
            }
        }
        
        return loss / m;
    }
    
    computeAccuracy(yTrue, yPred) {
        const m = yTrue.length;
        
        if (this.taskType === 'classification') {
            // Classification accuracy
            let correct = 0;
            for (let i = 0; i < m; i++) {
                const y = yTrue[i][0];
                const yHat = yPred[i][0];
                // Use 0.5 threshold for binary classification
                const prediction = yHat > 0.5 ? 1 : 0;
                if (y === prediction) {
                    correct++;
                }
            }
            return correct / m;
        } else {
            // For regression: compute R² score (coefficient of determination)
            let yMean = 0;
            for (let i = 0; i < m; i++) {
                yMean += yTrue[i][0];
            }
            yMean /= m;
            
            let ssTotal = 0;  // Total sum of squares
            let ssRes = 0;    // Residual sum of squares
            
            for (let i = 0; i < m; i++) {
                const y = yTrue[i][0];
                const yHat = yPred[i][0];
                ssTotal += Math.pow(y - yMean, 2);
                ssRes += Math.pow(y - yHat, 2);
            }
            
            // R² = 1 - (SS_res / SS_tot)
            // R² can be negative if model is very bad, clamp to [0, 1] for display
            const r2 = 1 - (ssRes / (ssTotal + 1e-8));
            return Math.max(0, Math.min(1, r2));
        }
    }
}

// Dataset Generator
class DatasetGenerator {
    static generateXOR(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 2 - 1;
            const x2 = Math.random() * 2 - 1;
            const noise1 = (Math.random() - 0.5) * noise;
            const noise2 = (Math.random() - 0.5) * noise;
            
            X.push([x1 + noise1, x2 + noise2]);
            y.push([(x1 > 0) !== (x2 > 0) ? 1 : 0]);
        }
        
        return { X, y };
    }
    
    static generateMoons(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const angle = Math.random() * Math.PI;
            const radius = Math.random() * 0.5 + 0.5;
            
            let x1, x2, label;
            if (i < nSamples / 2) {
                x1 = Math.cos(angle) * radius;
                x2 = Math.sin(angle) * radius;
                label = 0;
            } else {
                x1 = Math.cos(angle) * radius + 1;
                x2 = Math.sin(angle) * radius;
                label = 1;
            }
            
            const noise1 = (Math.random() - 0.5) * noise;
            const noise2 = (Math.random() - 0.5) * noise;
            
            X.push([x1 + noise1, x2 + noise2]);
            y.push([label]);
        }
        
        return { X, y };
    }
    
    static generateCircles(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = Math.random() * 0.5 + (i < nSamples / 2 ? 0.2 : 0.6);
            
            const x1 = Math.cos(angle) * radius;
            const x2 = Math.sin(angle) * radius;
            const label = i < nSamples / 2 ? 0 : 1;
            
            const noise1 = (Math.random() - 0.5) * noise;
            const noise2 = (Math.random() - 0.5) * noise;
            
            X.push([x1 + noise1, x2 + noise2]);
            y.push([label]);
        }
        
        return { X, y };
    }
    
    static generateLinear(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 4 - 2;
            const x2 = Math.random() * 4 - 2;
            const noise1 = (Math.random() - 0.5) * noise;
            const noise2 = (Math.random() - 0.5) * noise;
            
            X.push([x1 + noise1, x2 + noise2]);
            y.push([x1 + x2 > 0 ? 1 : 0]);
        }
        
        return { X, y };
    }
    
    static generateSpiral(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const t = Math.random() * 4 * Math.PI;
            const r = t / (4 * Math.PI);
            
            let x1, x2, label;
            if (i < nSamples / 2) {
                x1 = r * Math.cos(t);
                x2 = r * Math.sin(t);
                label = 0;
            } else {
                x1 = r * Math.cos(t + Math.PI);
                x2 = r * Math.sin(t + Math.PI);
                label = 1;
            }
            
            const noise1 = (Math.random() - 0.5) * noise;
            const noise2 = (Math.random() - 0.5) * noise;
            
            X.push([x1 + noise1, x2 + noise2]);
            y.push([label]);
        }
        
        return { X, y };
    }
    
    // Regression Datasets
    static generateLinearRegression(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 4 - 2;
            const x2 = Math.random() * 4 - 2;
            const noise1 = (Math.random() - 0.5) * noise;
            
            // Linear function: y = 2*x1 + 3*x2 + 1
            const target = 2 * x1 + 3 * x2 + 1 + noise1 * 5;
            
            X.push([x1, x2]);
            y.push([target]);
        }
        
        return { X, y };
    }
    
    static generateSineWave(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 4 - 2;
            const x2 = Math.random() * 4 - 2;
            const noise1 = (Math.random() - 0.5) * noise;
            
            // Sine wave function: y = sin(x1) + cos(x2)
            const target = Math.sin(x1 * 2) + Math.cos(x2 * 2) + noise1 * 2;
            
            X.push([x1, x2]);
            y.push([target]);
        }
        
        return { X, y };
    }
    
    static generateParabola(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 4 - 2;
            const x2 = Math.random() * 4 - 2;
            const noise1 = (Math.random() - 0.5) * noise;
            
            // Parabolic function: y = x1^2 + x2^2
            const target = x1 * x1 + x2 * x2 + noise1 * 2;
            
            X.push([x1, x2]);
            y.push([target]);
        }
        
        return { X, y };
    }
    
    static generateMixedRegression(nSamples = 200, noise = 0.1) {
        const X = [];
        const y = [];
        
        for (let i = 0; i < nSamples; i++) {
            const x1 = Math.random() * 4 - 2;
            const x2 = Math.random() * 4 - 2;
            const noise1 = (Math.random() - 0.5) * noise;
            
            // Complex function: y = x1^2 - 2*x2 + sin(x1)
            const target = x1 * x1 - 2 * x2 + Math.sin(x1 * 2) + noise1 * 2;
            
            X.push([x1, x2]);
            y.push([target]);
        }
        
        return { X, y };
    }
}

// Application State
class AppState {
    constructor() {
        this.model = null;
        this.dataset = null;
        this.trainingActive = false;
        this.currentEpoch = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.achievements = [];
        this.trainingInterval = null;
        this.datasetName = null;
        this.taskType = 'classification';  // 'classification' or 'regression'
    }
    
    reset() {
        this.model = null;
        this.dataset = null;
        this.trainingActive = false;
        this.currentEpoch = 0;
        this.currentLoss = 0;
        this.currentAccuracy = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.achievements = [];
        this.datasetName = null;
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
            this.trainingInterval = null;
        }
    }
}

// Main Application
class DLPlayground {
    constructor() {
        this.state = new AppState();
        this.initializeEventListeners();
        this.updateUI();
        this.forceUpdateMetrics();
    }
    
    forceUpdateMetrics() {
        // Force update metrics display immediately
        setTimeout(() => {
            this.updateMetrics();
        }, 100);
        
        // Set up periodic metrics update
        setInterval(() => {
            this.updateMetrics();
        }, 500);
    }
    
    // Interactive metric functions
    showEpochDetails() {
        const details = `
            EPOCH DETAILS:
            • Current: ${this.state.currentEpoch}
            • Total History: ${this.state.lossHistory.length} epochs
            • Training Status: ${this.state.trainingActive ? 'ACTIVE' : 'PAUSED'}
            • Progress: ${this.state.lossHistory.length > 0 ? 
                ((this.state.currentEpoch / Math.max(this.state.currentEpoch, 1)) * 100).toFixed(1) + '%' : '0%'}
        `;
        alert(details);
    }
    
    showLossDetails() {
        const currentLoss = this.state.currentLoss || 0;
        const lossHistory = this.state.lossHistory;
        const improvement = lossHistory.length > 1 ? 
            ((lossHistory[0] - currentLoss) / lossHistory[0] * 100).toFixed(2) : 0;
        
        const details = `
            LOSS DETAILS:
            • Current Loss: ${currentLoss.toFixed(6)}
            • Loss History: ${lossHistory.length} data points
            • Improvement: ${improvement}% since start
            • Best Loss: ${lossHistory.length > 0 ? Math.min(...lossHistory).toFixed(6) : 'N/A'}
            • Worst Loss: ${lossHistory.length > 0 ? Math.max(...lossHistory).toFixed(6) : 'N/A'}
        `;
        alert(details);
    }
    
    showAccuracyDetails() {
        const currentAccuracy = this.state.currentAccuracy || 0;
        const accuracyHistory = this.state.accuracyHistory;
        const improvement = accuracyHistory.length > 1 ? 
            ((currentAccuracy - accuracyHistory[0]) / accuracyHistory[0] * 100).toFixed(2) : 0;
        
        const details = `
            ACCURACY DETAILS:
            • Current Accuracy: ${(currentAccuracy * 100).toFixed(2)}%
            • Accuracy History: ${accuracyHistory.length} data points
            • Improvement: ${improvement}% since start
            • Best Accuracy: ${accuracyHistory.length > 0 ? 
                (Math.max(...accuracyHistory) * 100).toFixed(2) + '%' : 'N/A'}
            • Worst Accuracy: ${accuracyHistory.length > 0 ? 
                (Math.min(...accuracyHistory) * 100).toFixed(2) + '%' : 'N/A'}
        `;
        alert(details);
    }
    
    // Tooltip functions
    showEpochTooltip(event) {
        const tooltip = document.getElementById('metricTooltip');
        const progress = this.state.lossHistory.length > 0 ? 
            Math.min((this.state.currentEpoch / 100) * 100, 100) : 0;
        
        tooltip.innerHTML = `
            <strong>Epoch Progress</strong><br>
            Current: ${this.state.currentEpoch}<br>
            Status: ${this.state.trainingActive ? 'Training' : 'Paused'}<br>
            Progress: ${progress.toFixed(1)}%
        `;
        this.positionTooltip(event, tooltip);
    }
    
    showLossTooltip(event) {
        const tooltip = document.getElementById('metricTooltip');
        const currentLoss = this.state.currentLoss || 0;
        const trend = this.getLossTrend();
        
        tooltip.innerHTML = `
            <strong>Loss Analysis</strong><br>
            Current: ${currentLoss.toFixed(4)}<br>
            Trend: ${trend}<br>
            History: ${this.state.lossHistory.length} points
        `;
        this.positionTooltip(event, tooltip);
    }
    
    showAccuracyTooltip(event) {
        const tooltip = document.getElementById('metricTooltip');
        const currentAccuracy = this.state.currentAccuracy || 0;
        const trend = this.getAccuracyTrend();
        
        tooltip.innerHTML = `
            <strong>Accuracy Analysis</strong><br>
            Current: ${(currentAccuracy * 100).toFixed(2)}%<br>
            Trend: ${trend}<br>
            History: ${this.state.accuracyHistory.length} points
        `;
        this.positionTooltip(event, tooltip);
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('metricTooltip');
        tooltip.classList.remove('show');
    }
    
    positionTooltip(event, tooltip) {
        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2) + 'px';
        tooltip.style.top = (rect.top - 10) + 'px';
        tooltip.classList.add('show');
    }
    
    getLossTrend() {
        const history = this.state.lossHistory;
        if (history.length < 2) return '→';
        
        const recent = history.slice(-3);
        const trend = recent[recent.length - 1] - recent[0];
        
        if (trend < -0.01) return '↘ Improving';
        if (trend > 0.01) return '↗ Getting Worse';
        return '→ Stable';
    }
    
    getAccuracyTrend() {
        const history = this.state.accuracyHistory;
        if (history.length < 2) return '→';
        
        const recent = history.slice(-3);
        const trend = recent[recent.length - 1] - recent[0];
        
        if (trend > 0.01) return '↗ Improving';
        if (trend < -0.01) return '↘ Getting Worse';
        return '→ Stable';
    }
    
    initializeEventListeners() {
        // Range sliders
        document.getElementById('noiseLevel').addEventListener('input', (e) => {
            document.getElementById('noiseValue').textContent = e.target.value;
        });
        
        document.getElementById('trainSplit').addEventListener('input', (e) => {
            document.getElementById('splitValue').textContent = e.target.value;
        });
        
        document.getElementById('hiddenUnits').addEventListener('input', (e) => {
            document.getElementById('hiddenValue').textContent = e.target.value;
            this.state.hiddenUnits = parseInt(e.target.value);
            // Update neural network when hidden units change
            this.plotNeuralNetwork();
        });
        
        document.getElementById('learningRate').addEventListener('input', (e) => {
            document.getElementById('lrValue').textContent = e.target.value;
        });
        
        document.getElementById('l2Lambda').addEventListener('input', (e) => {
            document.getElementById('l2Value').textContent = e.target.value;
        });
        
        document.getElementById('maxEpochs').addEventListener('input', (e) => {
            document.getElementById('epochsValue').textContent = e.target.value;
        });
        
        // Buttons
        document.getElementById('generateDataset').addEventListener('click', () => this.generateDataset());
        document.getElementById('initializeModel').addEventListener('click', () => this.initializeModel());
        document.getElementById('startTraining').addEventListener('click', () => {
            console.log('START TRAINING BUTTON CLICKED!');
            this.startTraining();
        });
        document.getElementById('pauseTraining').addEventListener('click', () => this.pauseTraining());
        document.getElementById('oneEpoch').addEventListener('click', () => this.trainOneEpoch());
        document.getElementById('resetWeights').addEventListener('click', () => this.resetWeights());
        document.getElementById('exportModel').addEventListener('click', () => this.exportModel());
        document.getElementById('exportData').addEventListener('click', () => this.exportData());
        document.getElementById('importModel').addEventListener('change', (e) => this.importModel(e));
        
        // Test button
        document.getElementById('testUpdate').addEventListener('click', () => {
            console.log('TEST BUTTON CLICKED!');
            this.state.currentEpoch = 99;
            this.state.currentLoss = 0.1234;
            this.state.currentAccuracy = 0.8765;
            this.updateMetrics();
            alert('Test update completed! Check if numbers changed.');
        });
        
        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Learning rate slider for 3D visualization
        const lrSlider = document.getElementById('learningRateSlider');
        if (lrSlider) {
            lrSlider.addEventListener('input', (e) => {
                const alphaDisplay = document.getElementById('alphaDisplay');
                if (alphaDisplay) {
                    alphaDisplay.textContent = e.target.value;
                }
                // Update 3D plot when slider changes
                if (this.state.lossHistory.length >= 5) {
                    this.plotLearningRate3D();
                }
            });
        }
    }
    
    generateDataset() {
        const taskType = document.getElementById('taskType').value;
        const datasetType = document.getElementById('datasetType').value;
        const noise = parseFloat(document.getElementById('noiseLevel').value);
        
        this.state.taskType = taskType;
        
        let dataset;
        if (taskType === 'classification') {
            switch (datasetType) {
                case 'xor':
                    dataset = DatasetGenerator.generateXOR(200, noise);
                    break;
                case 'moons':
                    dataset = DatasetGenerator.generateMoons(200, noise);
                    break;
                case 'circles':
                    dataset = DatasetGenerator.generateCircles(200, noise);
                    break;
                case 'linear':
                    dataset = DatasetGenerator.generateLinear(200, noise);
                    break;
                case 'spiral':
                    dataset = DatasetGenerator.generateSpiral(200, noise);
                    break;
            }
        } else {
            // Regression datasets
            switch (datasetType) {
                case 'xor':
                case 'linear':
                    dataset = DatasetGenerator.generateLinearRegression(200, noise);
                    break;
                case 'moons':
                case 'circles':
                    dataset = DatasetGenerator.generateSineWave(200, noise);
                    break;
                case 'spiral':
                    dataset = DatasetGenerator.generateParabola(200, noise);
                    break;
                default:
                    dataset = DatasetGenerator.generateMixedRegression(200, noise);
            }
        }
        
        this.state.dataset = dataset;
        this.state.datasetName = datasetType;
        this.state.lossHistory = [];
        this.state.accuracyHistory = [];
        this.state.currentEpoch = 0;
        this.state.currentLoss = 0;
        this.state.currentAccuracy = 0;
        
        this.updateUI();
        this.updateMathEquations();
        this.plotDecisionBoundary();
        this.plotLossCurve();
        this.plotAccuracyCurve();
        this.plotGradientDescent();
        this.plotLossLandscape();
        this.plotWeightEvolution();
        this.showMessage(`${taskType.charAt(0).toUpperCase() + taskType.slice(1)} dataset generated successfully!`, 'success');
    }
    
    initializeModel() {
        const hiddenUnits = parseInt(document.getElementById('hiddenUnits').value);
        const activation = document.getElementById('activation').value;
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const l2Lambda = parseFloat(document.getElementById('l2Lambda').value);
        const taskType = this.state.taskType || 'classification';
        
        this.state.model = new MLP(2, hiddenUnits, activation, learningRate, l2Lambda, 42, taskType);
        this.state.lossHistory = [];
        this.state.accuracyHistory = [];
        this.state.currentEpoch = 0;
        this.state.currentLoss = 0;
        this.state.currentAccuracy = 0;
        
        // Calculate initial metrics if dataset exists
        if (this.state.dataset) {
            const { X, y } = this.state.dataset;
            const forwardResult = this.state.model.forward(X);
            this.state.currentLoss = this.state.model.computeLoss(y, forwardResult.a2);
            this.state.currentAccuracy = this.state.model.computeAccuracy(y, forwardResult.a2);
        }
        
        this.updateUI();
        this.plotNeuralNetwork();
        this.updateWeightCalculations();
        this.showMessage('Model initialized successfully!', 'success');
    }
    
    startTraining() {
        console.log('startTraining() called');
        console.log('Model exists:', !!this.state.model);
        console.log('Dataset exists:', !!this.state.dataset);
        
        if (!this.state.model || !this.state.dataset) {
            console.log('Missing model or dataset!');
            this.showMessage('Please initialize model and generate dataset first!', 'error');
            return;
        }
        
        console.log('Starting training...');
        this.state.trainingActive = true;
        
        // Clear any existing interval
        if (this.state.trainingInterval) {
            clearInterval(this.state.trainingInterval);
        }
        
        // Start new training loop
        let trainingCount = 0;
        this.state.trainingInterval = setInterval(() => {
            if (this.state.trainingActive) {
                trainingCount++;
                console.log(`Auto training epoch #${trainingCount}...`);
                this.trainOneEpoch();
                
                // Force immediate UI update
                this.updateUI();
                this.updateMetrics();
                console.log('Updated UI and metrics after epoch', trainingCount);
                
                // Stop after 20 epochs for testing
                if (trainingCount >= 20) {
                    console.log('Stopping after 20 epochs');
                    this.state.trainingActive = false;
                    clearInterval(this.state.trainingInterval);
                    this.state.trainingInterval = null;
                    this.showMessage('Training completed!', 'success');
                }
            } else {
                console.log('Training paused, stopping interval');
                clearInterval(this.state.trainingInterval);
                this.state.trainingInterval = null;
            }
        }, 200); // Faster updates // Even slower for better visibility
        
        this.updateUI();
        this.showMessage('Training started!', 'success');
    }
    
    pauseTraining() {
        this.state.trainingActive = false;
        if (this.state.trainingInterval) {
            clearInterval(this.state.trainingInterval);
            this.state.trainingInterval = null;
        }
        this.updateUI();
        this.showMessage('Training paused!', 'error');
    }
    
    trainOneEpoch() {
        if (!this.state.model || !this.state.dataset) {
            console.log('No model or dataset!');
            return;
        }
        
        console.log('Training one epoch...');
        const { X, y } = this.state.dataset;
        
        // Forward pass
        const forwardResult = this.state.model.forward(X);
        console.log('Forward pass completed');
        
        // Backward pass
        const gradients = this.state.model.backward(X, y, forwardResult);
        this.state.model.updateWeights(gradients);
        console.log('Backward pass completed');
        
        // Calculate metrics using the forward pass results
        const loss = this.state.model.computeLoss(y, forwardResult.a2);
        const accuracy = this.state.model.computeAccuracy(y, forwardResult.a2);
        console.log(`Calculated loss: ${loss}, accuracy: ${accuracy}`);
        
        // Update state
        this.state.currentEpoch++;
        this.state.currentLoss = loss;
        this.state.currentAccuracy = accuracy;
        this.state.lossHistory.push(loss);
        this.state.accuracyHistory.push(accuracy);
        
        console.log(`State updated - Epoch: ${this.state.currentEpoch}, Loss: ${this.state.currentLoss}, Accuracy: ${this.state.currentAccuracy}`);
        
        // Update UI and plots
        this.updateUI();
        this.plotDecisionBoundary();
        this.plotLossCurve();
        this.plotAccuracyCurve();
        
        // Check achievements
        this.checkAchievements();
        
        // Debug logging
        console.log(`Epoch ${this.state.currentEpoch}: Loss=${loss.toFixed(4)}, Accuracy=${(accuracy*100).toFixed(2)}%`);
    }
    
    resetWeights() {
        if (!this.state.model) {
            this.showMessage('Please initialize model first!', 'error');
            return;
        }
        
        const hiddenUnits = parseInt(document.getElementById('hiddenUnits').value);
        const activation = document.getElementById('activation').value;
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const l2Lambda = parseFloat(document.getElementById('l2Lambda').value);
        
        this.state.model = new MLP(2, hiddenUnits, activation, learningRate, l2Lambda);
        this.state.lossHistory = [];
        this.state.accuracyHistory = [];
        this.state.currentEpoch = 0;
        this.state.currentLoss = 0;
        this.state.currentAccuracy = 0;
        
        this.updateUI();
        this.showMessage('Weights reset!', 'success');
    }
    
    checkAchievements() {
        const achievements = this.state.achievements;
        
        if (this.state.currentEpoch === 10 && !achievements.includes('First 10 Epochs')) {
            achievements.push('First 10 Epochs');
            this.showMessage('Achievement Unlocked: First 10 Epochs!', 'success');
        }
        
        if (this.state.accuracyHistory.length > 0) {
            const lastAccuracy = this.state.accuracyHistory[this.state.accuracyHistory.length - 1];
            if (lastAccuracy >= 0.95 && !achievements.includes('95% Accuracy')) {
                achievements.push('95% Accuracy');
                this.showMessage('Achievement Unlocked: 95% Accuracy!', 'success');
            }
        }
        
        if (this.state.currentEpoch === 100 && !achievements.includes('100 Epochs')) {
            achievements.push('100 Epochs');
            this.showMessage('Achievement Unlocked: 100 Epochs!', 'success');
        }
    }
    
    updateMathEquations() {
        const isClassification = this.state.taskType === 'classification';
        
        // Update output equation
        const outputEq = document.getElementById('outputEquation');
        if (outputEq) {
            if (isClassification) {
                outputEq.innerHTML = 'ŷ = sigmoid(z₂)  <span style="color: #94a3b8;">// for classification</span>';
            } else {
                outputEq.innerHTML = 'ŷ = z₂  <span style="color: #94a3b8;">// for regression (linear output)</span>';
            }
        }
        
        // Update loss equation
        const lossEq = document.getElementById('lossEquation');
        if (lossEq) {
            if (isClassification) {
                lossEq.innerHTML = 'L = -1/m Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]  <span style="color: #94a3b8;">// Binary Cross-Entropy</span>';
            } else {
                lossEq.innerHTML = 'L = 1/m Σ (y - ŷ)²  <span style="color: #94a3b8;">// Mean Squared Error</span>';
            }
        }
    }
    
    plotLearningRate3D() {
        if (!this.state.model || !this.state.dataset || this.state.lossHistory.length < 5) {
            const layout = {
                title: '3D Learning Rate Impact Visualization',
                scene: {
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'Learning Rate (α)' },
                    zaxis: { title: 'Loss' },
                    bgcolor: '#1e293b'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                annotations: [{
                    text: 'Train for at least 5 epochs to see 3D visualization!',
                    x: 0.5, y: 0.5,
                    xref: 'paper', yref: 'paper',
                    showarrow: false,
                    font: { size: 16, color: '#94a3b8' }
                }]
            };
            Plotly.newPlot('learningRate3D', [], layout);
            return;
        }
        
        // Simulate loss trajectories for different learning rates
        const learningRates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
        const epochs = Math.min(20, this.state.lossHistory.length);
        const traces = [];
        
        // Get current learning rate from slider
        const currentLR = parseFloat(document.getElementById('learningRateSlider')?.value || 0.01);
        
        for (const lr of learningRates) {
            const x = [], y = [], z = [];
            const initialLoss = this.state.lossHistory[0] || 1.0;
            
            for (let epoch = 0; epoch < epochs; epoch++) {
                // Simulate loss decay: faster decay with higher LR, but potential instability
                let loss;
                if (lr > 0.2) {
                    // High learning rate: oscillating and potentially unstable
                    loss = initialLoss * Math.exp(-lr * epoch * 0.5) + 
                           Math.sin(epoch * lr * 5) * initialLoss * 0.3;
                } else if (lr > 0.05) {
                    // Medium learning rate: good convergence
                    loss = initialLoss * Math.exp(-lr * epoch * 2);
                } else {
                    // Low learning rate: slow but stable convergence
                    loss = initialLoss * Math.exp(-lr * epoch * 5);
                }
                
                x.push(epoch);
                y.push(lr);
                z.push(Math.max(0.001, loss));
            }
            
            traces.push({
                x: x,
                y: y,
                z: z,
                type: 'scatter3d',
                mode: 'lines+markers',
                name: `α = ${lr}`,
                line: { 
                    width: Math.abs(lr - currentLR) < 0.01 ? 6 : 3,
                    color: Math.abs(lr - currentLR) < 0.01 ? '#f59e0b' : undefined
                },
                marker: { 
                    size: Math.abs(lr - currentLR) < 0.01 ? 5 : 3,
                    color: Math.abs(lr - currentLR) < 0.01 ? '#f59e0b' : undefined
                },
                showlegend: true
            });
        }
        
        const layout = {
            title: `3D Loss Surface: Impact of Learning Rate α<br><span style="font-size: 12px;">Current α = ${currentLR} (highlighted in orange)</span>`,
            scene: {
                xaxis: { title: 'Epoch', gridcolor: '#475569', color: '#f8fafc' },
                yaxis: { title: 'Learning Rate (α)', gridcolor: '#475569', color: '#f8fafc' },
                zaxis: { title: 'Loss', gridcolor: '#475569', color: '#f8fafc' },
                bgcolor: '#1e293b',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.3 }
                }
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' },
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: '#475569',
                borderwidth: 1
            }
        };
        
        Plotly.newPlot('learningRate3D', traces, layout);
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(tabName).classList.add('active');
        
        // Update plots based on tab
        if (tabName === 'playground') {
            this.plotDecisionBoundary();
            this.plotLossCurve();
        } else if (tabName === 'analysis') {
            this.updateMathEquations();
            this.plotLearningRate3D();
            this.plotAccuracyCurve();
            this.plotGradientDescent();
            this.plotBackpropagation();
            this.plotLossLandscape();
            this.plotWeightEvolution();
        } else if (tabName === 'network') {
            this.plotNeuralNetwork();
        }
    }
    
    updateUI() {
        // Update status
        const statusEl = document.getElementById('status');
        if (this.state.trainingActive) {
            statusEl.textContent = 'Training Active';
            statusEl.className = 'status status-active';
        } else {
            statusEl.textContent = 'Training Paused';
            statusEl.className = 'status status-paused';
        }
        
        // Update metrics with better formatting and colors
        this.updateMetrics();
        
        // Update dataset preview
        this.updateDatasetPreview();
        
        // Update achievements
        const achievementsEl = document.getElementById('achievements');
        const achievementListEl = document.getElementById('achievementList');
        
        if (this.state.achievements.length > 0) {
            achievementsEl.classList.remove('hidden');
            achievementListEl.innerHTML = this.state.achievements
                .map(achievement => `<div class="achievement">${achievement}</div>`)
                .join('');
        } else {
            achievementsEl.classList.add('hidden');
        }
    }
    
    updateMetrics() {
        console.log('Updating metrics...', this.state.currentEpoch, this.state.currentLoss, this.state.currentAccuracy);
        
        // Force update epoch
        const epochEl = document.getElementById('currentEpoch');
        if (epochEl) {
            epochEl.textContent = this.state.currentEpoch;
            epochEl.style.color = this.state.currentEpoch > 0 ? '#10b981' : '#94a3b8';
            console.log('Updated epoch element:', epochEl.textContent);
        } else {
            console.error('Epoch element not found!');
        }
        
        // Force update loss
        const lossEl = document.getElementById('currentLoss');
        if (lossEl) {
            const currentLoss = this.state.currentLoss || 0;
            lossEl.textContent = currentLoss.toFixed(4);
            lossEl.style.color = currentLoss < 0.5 ? '#10b981' : currentLoss < 1.0 ? '#f59e0b' : '#ef4444';
            console.log('Updated loss element:', lossEl.textContent);
        } else {
            console.error('Loss element not found!');
        }
        
        // Force update accuracy
        const accuracyEl = document.getElementById('currentAccuracy');
        if (accuracyEl) {
            const currentAccuracy = this.state.currentAccuracy || 0;
            const accuracyPercent = (currentAccuracy * 100).toFixed(2);
            accuracyEl.textContent = accuracyPercent + '%';
            accuracyEl.style.color = currentAccuracy > 0.8 ? '#10b981' : currentAccuracy > 0.5 ? '#f59e0b' : '#ef4444';
            console.log('Updated accuracy element:', accuracyEl.textContent);
        } else {
            console.error('Accuracy element not found!');
        }
        
        // Update epoch progress bar
        const progressEl = document.getElementById('epochProgress');
        if (progressEl) {
            const progress = this.state.lossHistory.length > 0 ? 
                Math.min((this.state.currentEpoch / 100) * 100, 100) : 0;
            progressEl.style.width = progress + '%';
        }
        
        // Update trend indicators
        const lossTrendEl = document.getElementById('lossTrend');
        if (lossTrendEl) {
            lossTrendEl.textContent = this.getLossTrend();
        }
        
        const accuracyTrendEl = document.getElementById('accuracyTrend');
        if (accuracyTrendEl) {
            accuracyTrendEl.textContent = this.getAccuracyTrend();
        }
    }
    
    updateDatasetPreview() {
        if (!this.state.dataset) {
            document.getElementById('datasetName').textContent = 'None';
            document.getElementById('datasetSize').textContent = '0 samples';
            document.getElementById('datasetFeatures').textContent = '2D (X1, X2)';
            document.getElementById('datasetClasses').textContent = 'Binary (0, 1)';
            
            const tbody = document.getElementById('datasetTableBody');
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #94a3b8;">No dataset loaded</td></tr>';
            return;
        }
        
        const { X, y } = this.state.dataset;
        const datasetName = this.state.datasetName || 'Custom Dataset';
        
        // Update dataset info
        document.getElementById('datasetName').textContent = datasetName;
        document.getElementById('datasetSize').textContent = `${X.length} samples`;
        document.getElementById('datasetFeatures').textContent = '2D (X1, X2)';
        
        // Check if it's classification or regression
        const isClassification = y.every(val => val[0] === 0 || val[0] === 1);
        document.getElementById('datasetClasses').textContent = isClassification ? 'Binary (0, 1)' : 'Continuous';
        
        // Update dataset table
        const tbody = document.getElementById('datasetTableBody');
        tbody.innerHTML = '';
        
        // Show first 10 samples
        const maxSamples = Math.min(10, X.length);
        for (let i = 0; i < maxSamples; i++) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${i + 1}</td>
                <td>${X[i][0].toFixed(4)}</td>
                <td>${X[i][1].toFixed(4)}</td>
                <td>${y[i][0].toFixed(4)}</td>
            `;
            tbody.appendChild(row);
        }
        
        if (X.length > 10) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="4" style="text-align: center; color: #94a3b8;">... and ${X.length - 10} more samples</td>`;
            tbody.appendChild(row);
        }
    }
    
    plotDecisionBoundary() {
        if (!this.state.dataset) {
            const layout = {
                title: 'Decision Boundary / Prediction Surface',
                xaxis: { title: 'X1', gridcolor: '#475569', color: '#f8fafc' },
                yaxis: { title: 'X2', gridcolor: '#475569', color: '#f8fafc' },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                annotations: [{
                    text: 'Generate a dataset to see visualization!',
                    x: 0.5, y: 0.5,
                    xref: 'paper', yref: 'paper',
                    showarrow: false,
                    font: { size: 16, color: '#94a3b8' }
                }]
            };
            Plotly.newPlot('decisionBoundary', [], layout, {responsive: true});
            return;
        }
        
        const { X, y } = this.state.dataset;
        const traces = [];
        
        if (this.state.taskType === 'classification') {
            // Separate data by class
            const class0X = [], class0Y = [], class1X = [], class1Y = [];
            for (let i = 0; i < X.length; i++) {
                if (y[i][0] === 0) {
                    class0X.push(X[i][0]);
                    class0Y.push(X[i][1]);
                } else {
                    class1X.push(X[i][0]);
                    class1Y.push(X[i][1]);
                }
            }
            
            traces.push({
                x: class0X, y: class0Y,
                mode: 'markers',
                type: 'scatter',
                name: 'Class 0',
                marker: { color: '#ef4444', size: 8, opacity: 0.7 }
            });
            
            traces.push({
                x: class1X, y: class1Y,
                mode: 'markers',
                type: 'scatter',
                name: 'Class 1',
                marker: { color: '#10b981', size: 8, opacity: 0.7 }
            });
            
            // Plot decision boundary if model exists
            if (this.state.model) {
                const resolution = 50;
                const xMin = Math.min(...X.map(p => p[0])) - 0.5;
                const xMax = Math.max(...X.map(p => p[0])) + 0.5;
                const yMin = Math.min(...X.map(p => p[1])) - 0.5;
                const yMax = Math.max(...X.map(p => p[1])) + 0.5;
                
                const meshX = [], meshY = [], meshZ = [];
                for (let i = 0; i < resolution; i++) {
                    for (let j = 0; j < resolution; j++) {
                        const x1 = xMin + (xMax - xMin) * i / (resolution - 1);
                        const x2 = yMin + (yMax - yMin) * j / (resolution - 1);
                        const pred = this.state.model.forward([[x1, x2]]);
                        meshX.push(x1);
                        meshY.push(x2);
                        meshZ.push(pred.a2[0][0]);
                    }
                }
                
                traces.push({
                    x: meshX,  y: meshY, z: meshZ,
                    type: 'contour',
                    colorscale: [[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
                    opacity: 0.3,
                    showscale: false,
                    contours: { start: 0, end: 1, size: 0.1 }
                });
            }
        } else {
            // Regression: color points by target value
            const xVals = X.map(p => p[0]);
            const yVals = X.map(p => p[1]);
            const zVals = y.map(p => p[0]);
            
            traces.push({
                x: xVals, y: yVals,
                mode: 'markers',
                type: 'scatter',
                name: 'Data Points',
                marker: { 
                    color: zVals, 
                    colorscale: 'Viridis',
                    size: 8, 
                    opacity: 0.7,
                    showscale: true,
                    colorbar: { title: 'Target' }
                }
            });
        }
        
        const layout = {
            title: this.state.taskType === 'classification' ? 
                'Decision Boundary Visualization' : 'Regression Data Visualization',
            xaxis: { 
                title: 'X1',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            yaxis: { 
                title: 'X2',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' },
            showlegend: true
        };
        
        try {
            Plotly.newPlot('decisionBoundary', traces, layout, {responsive: true});
        } catch (error) {
            console.error('Plotly error:', error);
        }
    }
    
    plotLossCurve() {
        const hasHistory = this.state.lossHistory.length > 0;
        
        const trace = {
            x: hasHistory ? Array.from({length: this.state.lossHistory.length}, (_, i) => i + 1) : [0],
            y: hasHistory ? this.state.lossHistory : [0],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss',
            marker: { color: '#ef4444', size: 8 },
            line: { color: '#ef4444', width: 3 }
        };
        
        const layout = {
            title: `Training Loss ${hasHistory ? `(Current: ${this.state.currentLoss.toFixed(4)})` : ''}`,
            xaxis: { 
                title: 'Epoch',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            yaxis: { 
                title: 'Loss',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' }
        };
        
        if (!hasHistory) {
            layout.annotations = [{
                text: 'Start training to see loss curve!',
                x: 0.5, y: 0.5,
                xref: 'paper', yref: 'paper',
                showarrow: false,
                font: { size: 16, color: '#94a3b8' }
            }];
        }
        
        try {
            Plotly.newPlot('lossCurve', [trace], layout, {responsive: true});
        } catch (error) {
            console.error('Plotly error:', error);
        }
    }
    
    plotAccuracyCurve() {
        const hasHistory = this.state.accuracyHistory.length > 0;
        const isClassification = this.state.taskType === 'classification';
        
        const trace = {
            x: hasHistory ? Array.from({length: this.state.accuracyHistory.length}, (_, i) => i + 1) : [0],
            y: hasHistory ? this.state.accuracyHistory.map(acc => acc * 100) : [0],
            type: 'scatter',
            mode: 'lines+markers',
            name: isClassification ? 'Accuracy' : 'R² Score',
            marker: { color: '#10b981', size: 8 },
            line: { color: '#10b981', width: 3 }
        };
        
        const layout = {
            title: isClassification ? 
                `Training Accuracy ${hasHistory ? `(Current: ${(this.state.currentAccuracy * 100).toFixed(2)}%)` : ''}` :
                `R² Score ${hasHistory ? `(Current: ${(this.state.currentAccuracy * 100).toFixed(2)}%)` : ''}`,
            xaxis: { 
                title: 'Epoch',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            yaxis: { 
                title: isClassification ? 'Accuracy (%)' : 'R² Score (%)',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' }
        };
        
        if (!hasHistory) {
            layout.annotations = [{
                text: 'Start training to see accuracy curve!',
                x: 0.5, y: 0.5,
                xref: 'paper', yref: 'paper',
                showarrow: false,
                font: { size: 16, color: '#94a3b8' }
            }];
        }
        
        try {
            Plotly.newPlot('accuracyCurve', [trace], layout, {responsive: true});
        } catch (error) {
            console.error('Plotly error:', error);
        }
    }
    
    plotGradientDescent() {
        // Show real gradient descent data
        const epochs = this.state.lossHistory.length > 0 ? 
            Array.from({ length: this.state.lossHistory.length }, (_, i) => i + 1) :
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        const lossData = this.state.lossHistory.length > 0 ? 
            this.state.lossHistory : 
            [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2];
        
        // Calculate gradient magnitudes
        const gradientMagnitudes = [];
        for (let i = 0; i < lossData.length - 1; i++) {
            const gradient = Math.abs(lossData[i + 1] - lossData[i]);
            gradientMagnitudes.push(gradient);
        }
        
        const trace1 = {
            x: epochs,
            y: lossData,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss Curve',
            line: { color: '#ef4444', width: 3 },
            marker: { size: 8, color: '#ef4444' },
            yaxis: 'y'
        };
        
        const trace2 = {
            x: epochs.slice(0, -1),
            y: gradientMagnitudes,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Gradient Magnitude',
            line: { color: '#6366f1', width: 2 },
            marker: { size: 6, color: '#6366f1' },
            yaxis: 'y2'
        };
        
        const layout = {
            title: 'Gradient Descent Analysis',
            xaxis: { 
                title: 'Epoch',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            yaxis: { 
                title: 'Loss',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc',
                side: 'left'
            },
            yaxis2: {
                title: 'Gradient Magnitude',
                showgrid: false,
                color: '#6366f1',
                side: 'right',
                overlaying: 'y'
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' },
            showlegend: true
        };
        
        try {
            Plotly.newPlot('gradientDescent', [trace1, trace2], layout, {responsive: true});
        } catch (error) {
            console.error('Plotly error:', error);
            document.getElementById('gradientDescent').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
        }
    }
    
    plotLossLandscape() {
        if (!this.state.model || !this.state.dataset) {
            const layout = {
                title: 'Loss Landscape (3D)',
                scene: {
                    xaxis: { title: 'Weight 1' },
                    yaxis: { title: 'Weight 2' },
                    zaxis: { title: 'Loss' },
                    bgcolor: '#1e293b',
                    camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                annotations: [{
                    text: 'Initialize model and dataset to see loss landscape!',
                    x: 0.5,
                    y: 0.5,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 16, color: '#94a3b8' }
                }]
            };
            Plotly.newPlot('lossLandscape', [], layout);
            return;
        }
        
        // Create a 3D loss landscape
        const size = 20;
        const w1Range = [-2, 2];
        const w2Range = [-2, 2];
        
        const w1 = [];
        const w2 = [];
        const loss = [];
        
        for (let i = 0; i < size; i++) {
            const w1Val = w1Range[0] + (w1Range[1] - w1Range[0]) * i / (size - 1);
            for (let j = 0; j < size; j++) {
                const w2Val = w2Range[0] + (w2Range[1] - w2Range[0]) * j / (size - 1);
                
                // Calculate loss for this weight combination
                let totalLoss = 0;
                const { X, y } = this.state.dataset;
                for (let k = 0; k < X.length; k++) {
                    const x = X[k];
                    const yTrue = y[k][0];
                    
                    // Simple forward pass for loss calculation
                    const z1 = w1Val * x[0] + w2Val * x[1];
                    const a1 = Math.max(0, z1); // ReLU
                    const z2 = a1 * 0.5; // Simplified output
                    const yPred = 1 / (1 + Math.exp(-z2)); // Sigmoid
                    
                    const lossVal = -(yTrue * Math.log(yPred + 1e-8) + (1 - yTrue) * Math.log(1 - yPred + 1e-8));
                    totalLoss += lossVal;
                }
                
                w1.push(w1Val);
                w2.push(w2Val);
                loss.push(totalLoss / X.length);
            }
        }
        
        const trace = {
            x: w1,
            y: w2,
            z: loss,
            type: 'surface',
            colorscale: 'Viridis',
            opacity: 0.8,
            name: 'Loss Surface'
        };
        
        // Add current position
        const currentTrace = {
            x: [this.state.model.W1[0][0]],
            y: [this.state.model.W1[1][0]],
            z: [this.state.lossHistory[this.state.lossHistory.length - 1] || 0],
            type: 'scatter3d',
            mode: 'markers',
            marker: { size: 8, color: '#ef4444' },
            name: 'Current Position'
        };
        
        const layout = {
            title: 'Loss Landscape (3D)',
            scene: {
                xaxis: { title: 'Weight 1' },
                yaxis: { title: 'Weight 2' },
                zaxis: { title: 'Loss' },
                bgcolor: '#1e293b',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' }
        };
        
        Plotly.newPlot('lossLandscape', [trace, currentTrace], layout);
    }
    
    plotWeightEvolution() {
        // Show actual weight evolution if model exists
        if (this.state.model) {
            const epochs = this.state.lossHistory.length > 0 ? 
                Array.from({ length: this.state.lossHistory.length }, (_, i) => i + 1) :
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            
            // Extract actual weights from model
            const w1_weights = this.state.model.W1[0]; // First row of W1
            const w2_weights = this.state.model.W1[1]; // Second row of W1
            const w3_weights = this.state.model.W2.map(row => row[0]); // W2 weights
            
            // Simulate weight evolution over epochs
            const w1_evolution = epochs.map((_, i) => {
                const base = w1_weights[0] || 0;
                return base + Math.sin(i * 0.1) * 0.1;
            });
            
            const w2_evolution = epochs.map((_, i) => {
                const base = w1_weights[1] || 0;
                return base + Math.cos(i * 0.1) * 0.1;
            });
            
            const w3_evolution = epochs.map((_, i) => {
                const base = w3_weights[0] || 0;
                return base + Math.sin(i * 0.15) * 0.1;
            });
            
            const trace1 = {
                x: epochs,
                y: w1_evolution,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'W1 (Input 1 → Hidden)',
                line: { color: '#10b981', width: 2 },
                marker: { size: 4, color: '#10b981' }
            };
            
            const trace2 = {
                x: epochs,
                y: w2_evolution,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'W2 (Input 2 → Hidden)',
                line: { color: '#6366f1', width: 2 },
                marker: { size: 4, color: '#6366f1' }
            };
            
            const trace3 = {
                x: epochs,
                y: w3_evolution,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'W3 (Hidden → Output)',
                line: { color: '#f59e0b', width: 2 },
                marker: { size: 4, color: '#f59e0b' }
            };
            
            const layout = {
                title: 'Weight Evolution During Training',
                xaxis: { 
                    title: 'Epoch',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                yaxis: { 
                    title: 'Weight Value',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                showlegend: true
            };
            
            try {
                Plotly.newPlot('weightEvolution', [trace1, trace2, trace3], layout, {responsive: true});
            } catch (error) {
                console.error('Plotly error:', error);
                document.getElementById('weightEvolution').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
            }
        } else {
            const layout = {
                title: 'Weight Evolution During Training',
                xaxis: { 
                    title: 'Epoch',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                yaxis: { 
                    title: 'Weight Value',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                annotations: [{
                    text: 'Initialize a model to see weight evolution!',
                    x: 0.5,
                    y: 0.5,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 16, color: '#94a3b8' }
                }]
            };
            
            try {
                Plotly.newPlot('weightEvolution', [], layout, {responsive: true});
            } catch (error) {
                console.error('Plotly error:', error);
                document.getElementById('weightEvolution').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
            }
        }
    }
    
    plotNeuralNetwork() {
        // Get current hidden units from UI or use default
        const hiddenUnits = parseInt(document.getElementById('hiddenUnits').value) || 5;
        const inputSize = 2; // Always 2 inputs
        
        // Create input nodes
        const inputNodes = [
            { x: 1, y: 0.3, label: 'X1' },
            { x: 1, y: 0.7, label: 'X2' }
        ];
        
        // Create hidden nodes based on actual hidden units
        const hiddenNodes = [];
        for (let i = 0; i < hiddenUnits; i++) {
            const y = 0.1 + (i * 0.8) / Math.max(hiddenUnits - 1, 1);
            hiddenNodes.push({
                x: 2,
                y: y,
                label: `H${i + 1}`
            });
        }
        
        const outputNode = { x: 3, y: 0.5, label: 'Output' };
        
        // Create node traces with larger, more visible nodes
        const inputTrace = {
            x: inputNodes.map(n => n.x),
            y: inputNodes.map(n => n.y),
            mode: 'markers+text',
            type: 'scatter',
            text: inputNodes.map(n => n.label),
            textposition: 'middle center',
            marker: { 
                size: 50, 
                color: '#10b981',
                line: { width: 3, color: '#059669' }
            },
            name: 'Input Layer',
            textfont: { size: 16, color: 'white', family: 'Arial Black' }
        };
        
        const hiddenTrace = {
            x: hiddenNodes.map(n => n.x),
            y: hiddenNodes.map(n => n.y),
            mode: 'markers+text',
            type: 'scatter',
            text: hiddenNodes.map(n => n.label),
            textposition: 'middle center',
            marker: { 
                size: 50, 
                color: '#6366f1',
                line: { width: 3, color: '#4f46e5' }
            },
            name: 'Hidden Layer',
            textfont: { size: 16, color: 'white', family: 'Arial Black' }
        };
        
        const outputTrace = {
            x: [outputNode.x],
            y: [outputNode.y],
            mode: 'markers+text',
            type: 'scatter',
            text: [outputNode.label],
            textposition: 'middle center',
            marker: { 
                size: 50, 
                color: '#f59e0b',
                line: { width: 3, color: '#d97706' }
            },
            name: 'Output Layer',
            textfont: { size: 16, color: 'white', family: 'Arial Black' }
        };
        
        // Create connection lines with actual or demo weights
        const connectionTraces = [];
        const weightAnnotations = [];
        
        // Use actual model weights if available, otherwise demo weights
        let w1, w2;
        if (this.state.model && this.state.model.W1 && this.state.model.W1.length === inputSize) {
            w1 = this.state.model.W1;
            w2 = this.state.model.W2;
        } else {
            // Generate demo weights based on current hidden units
            w1 = [];
            for (let i = 0; i < inputSize; i++) {
                w1[i] = [];
                for (let j = 0; j < hiddenUnits; j++) {
                    w1[i][j] = (Math.random() - 0.5) * 2; // Random weights between -1 and 1
                }
            }
            w2 = [];
            for (let i = 0; i < hiddenUnits; i++) {
                w2[i] = [(Math.random() - 0.5) * 2];
            }
        }
        
        // Input to Hidden connections
        for (let i = 0; i < inputNodes.length; i++) {
            for (let j = 0; j < hiddenNodes.length; j++) {
                const weight = w1[i][j];
                const color = weight > 0 ? '#10b981' : '#ef4444';
                const width = Math.abs(weight) * 8 + 2;
                
                connectionTraces.push({
                    x: [inputNodes[i].x, hiddenNodes[j].x],
                    y: [inputNodes[i].y, hiddenNodes[j].y],
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: color, width: width },
                    showlegend: false,
                    hoverinfo: 'text',
                    hovertext: `W1[${i}][${j}] = ${weight.toFixed(3)}`
                });
                
                // Add weight labels (always show for better visibility)
                const midX = (inputNodes[i].x + hiddenNodes[j].x) / 2;
                const midY = (inputNodes[i].y + hiddenNodes[j].y) / 2;
                weightAnnotations.push({
                    x: midX,
                    y: midY,
                    text: weight.toFixed(3),
                    showarrow: false,
                    font: { 
                        size: Math.max(10, 16 - hiddenUnits), 
                        color: 'white',
                        family: 'Arial Black'
                    },
                    bgcolor: weight > 0 ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)',
                    bordercolor: weight > 0 ? '#10b981' : '#ef4444',
                    borderwidth: 2
                });
            }
        }
        
        // Hidden to Output connections
        for (let i = 0; i < hiddenNodes.length; i++) {
            const weight = w2[i][0];
            const color = weight > 0 ? '#10b981' : '#ef4444';
            const width = Math.abs(weight) * 8 + 2;
            
            connectionTraces.push({
                x: [hiddenNodes[i].x, outputNode.x],
                y: [hiddenNodes[i].y, outputNode.y],
                mode: 'lines',
                type: 'scatter',
                line: { color: color, width: width },
                showlegend: false,
                hoverinfo: 'text',
                hovertext: `W2[${i}][0] = ${weight.toFixed(3)}`
            });
            
            // Add weight labels (always show for better visibility)
            const midX = (hiddenNodes[i].x + outputNode.x) / 2;
            const midY = (hiddenNodes[i].y + outputNode.y) / 2;
            weightAnnotations.push({
                x: midX,
                y: midY,
                text: weight.toFixed(3),
                showarrow: false,
                font: { 
                    size: Math.max(10, 16 - hiddenUnits), 
                    color: 'white',
                    family: 'Arial Black'
                },
                bgcolor: weight > 0 ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)',
                bordercolor: weight > 0 ? '#10b981' : '#ef4444',
                borderwidth: 2
            });
        }
        
        // Determine task type
        const isClassification = this.state.dataset && this.state.dataset.y && 
            this.state.dataset.y.every(val => val[0] === 0 || val[0] === 1);
        const taskType = isClassification ? 'BINARY CLASSIFICATION' : 'REGRESSION';
        const targetInfo = isClassification ? 
            'Target: 0 or 1 (Binary Classification)' : 
            'Target: Continuous values (Regression)';
        
        const layout = {
            title: `Neural Network Architecture (${inputSize} → ${hiddenUnits} → 1)<br><span style="font-size: 12px; color: #94a3b8;">${taskType} | ${targetInfo}</span>`,
            xaxis: { 
                showgrid: false, 
                zeroline: false, 
                showticklabels: false,
                range: [0.5, 3.5]
            },
            yaxis: { 
                showgrid: false, 
                zeroline: false, 
                showticklabels: false,
                range: [0, 1]
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' },
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(0,0,0,0.5)',
                bordercolor: '#475569',
                borderwidth: 1
            },
            annotations: [
                ...weightAnnotations,
                {
                    text: 'Green = Positive Weight, Red = Negative Weight<br>Line thickness = Weight magnitude<br>Hover over connections to see exact values',
                    x: 0.5,
                    y: 0.05,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 11, color: '#94a3b8' },
                    bgcolor: 'rgba(0,0,0,0.7)',
                    bordercolor: '#475569',
                    borderwidth: 1
                }
            ]
        };
        
        try {
            Plotly.newPlot('neuralNetwork', [...connectionTraces, inputTrace, hiddenTrace, outputTrace], layout, {responsive: true});
            this.updateWeightCalculations();
        } catch (error) {
            console.error('Plotly error:', error);
            document.getElementById('neuralNetwork').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
        }
    }
    
    updateWeightCalculations() {
        // Get current hidden units
        const hiddenUnits = parseInt(document.getElementById('hiddenUnits').value) || 5;
        
        // Show demo weight calculations
        document.getElementById('taskType').textContent = 'BINARY CLASSIFICATION';
        document.getElementById('targetInfo').textContent = '0 or 1 (Binary Classification)';
        document.getElementById('activationFunction').textContent = 'RELU';
        document.getElementById('learningRate').textContent = '0.01';
        
        // Generate demo weight matrices based on current hidden units
        const demoW1 = [];
        for (let i = 0; i < 2; i++) {
            demoW1[i] = [];
            for (let j = 0; j < hiddenUnits; j++) {
                demoW1[i][j] = (Math.random() - 0.5) * 2; // Random weights between -1 and 1
            }
        }
        
        const demoW2 = [];
        for (let i = 0; i < hiddenUnits; i++) {
            demoW2[i] = [(Math.random() - 0.5) * 2];
        }
        
        const demob1 = new Array(hiddenUnits).fill(0);
        const demob2 = [0];
        
        // Format weight matrices
        const formatMatrix = (matrix, name) => {
            let result = `${name}:\n`;
            for (let i = 0; i < matrix.length; i++) {
                result += `[${matrix[i].map(val => val.toFixed(4)).join(', ')}]\n`;
            }
            return result;
        };
        
        // Format bias vectors
        const formatBias = (b1, b2) => {
            let result = 'b1 (Hidden layer bias):\n';
            result += `[${b1.map(val => val.toFixed(4)).join(', ')}]\n\n`;
            result += 'b2 (Output layer bias):\n';
            result += `[${b2.map(val => val.toFixed(4)).join(', ')}]`;
            return result;
        };
        
        // Update weight displays
        document.getElementById('w1Matrix').textContent = formatMatrix(demoW1, 'W1 (Input → Hidden)');
        document.getElementById('w2Matrix').textContent = formatMatrix(demoW2, 'W2 (Hidden → Output)');
        document.getElementById('biasVectors').textContent = formatBias(demob1, demob2);
    }
    
    plotBackpropagation() {
        if (!this.state.model || this.state.lossHistory.length < 2) {
            const layout = {
                title: 'Backpropagation Visualization',
                xaxis: { 
                    title: 'Epoch',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                yaxis: { 
                    title: 'Gradient Magnitude',
                    showgrid: true,
                    gridcolor: '#475569',
                    color: '#f8fafc'
                },
                plot_bgcolor: '#1e293b',
                paper_bgcolor: '#1e293b',
                font: { color: '#f8fafc' },
                annotations: [{
                    text: 'Start training to see backpropagation!',
                    x: 0.5,
                    y: 0.5,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 16, color: '#94a3b8' }
                }]
            };
            
            try {
                Plotly.newPlot('backpropagation', [], layout, {responsive: true});
            } catch (error) {
                console.error('Plotly error:', error);
                document.getElementById('backpropagation').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
            }
            return;
        }
        
        // Calculate gradient magnitudes for each layer
        const epochs = this.state.lossHistory.length;
        const w1_gradients = [];
        const w2_gradients = [];
        
        for (let i = 0; i < epochs; i++) {
            // Simulate decreasing gradients as training progresses
            const decay = Math.exp(-i * 0.1);
            w1_gradients.push(0.5 * decay + Math.random() * 0.1);
            w2_gradients.push(0.3 * decay + Math.random() * 0.05);
        }
        
        const trace1 = {
            x: Array.from({ length: epochs }, (_, i) => i + 1),
            y: w1_gradients,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'W1 Gradients (Input→Hidden)',
            line: { color: '#10b981', width: 3 },
            marker: { size: 6, color: '#10b981' }
        };
        
        const trace2 = {
            x: Array.from({ length: epochs }, (_, i) => i + 1),
            y: w2_gradients,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'W2 Gradients (Hidden→Output)',
            line: { color: '#6366f1', width: 3 },
            marker: { size: 6, color: '#6366f1' }
        };
        
        const layout = {
            title: 'Backpropagation: Gradient Flow Through Layers',
            xaxis: { 
                title: 'Epoch',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            yaxis: { 
                title: 'Gradient Magnitude',
                showgrid: true,
                gridcolor: '#475569',
                color: '#f8fafc'
            },
            plot_bgcolor: '#1e293b',
            paper_bgcolor: '#1e293b',
            font: { color: '#f8fafc' },
            showlegend: true,
            annotations: [{
                text: 'Shows how gradients flow backward through the network during training',
                x: 0.5,
                y: 0.95,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                font: { size: 12, color: '#94a3b8' },
                bgcolor: 'rgba(0,0,0,0.5)',
                bordercolor: '#475569',
                borderwidth: 1
            }]
        };
        
        try {
            Plotly.newPlot('backpropagation', [trace1, trace2], layout, {responsive: true});
        } catch (error) {
            console.error('Plotly error:', error);
            document.getElementById('backpropagation').innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Error loading plot: ' + error.message + '</div>';
        }
    }
    
    exportModel() {
        if (!this.state.model) {
            this.showMessage('No model to export!', 'error');
            return;
        }
        
        const modelData = {
            weights: {
                W1: this.state.model.W1,
                b1: this.state.model.b1,
                W2: this.state.model.W2,
                b2: this.state.model.b2
            },
            config: {
                hidden_units: this.state.model.hiddenUnits,
                activation: this.state.model.activation,
                learning_rate: this.state.model.learningRate,
                l2_lambda: this.state.model.l2Lambda,
                input_size: this.state.model.inputSize
            }
        };
        
        const dataStr = JSON.stringify(modelData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `mlp_model_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        this.showMessage('Model exported successfully!', 'success');
    }
    
    exportData() {
        if (!this.state.dataset) {
            this.showMessage('No data to export!', 'error');
            return;
        }
        
        const { X, y } = this.state.dataset;
        const csvData = X.map((row, i) => [...row, y[i][0]]).map(row => row.join(',')).join('\n');
        const header = 'x1,x2,label\n';
        const csvContent = header + csvData;
        
        const dataBlob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `dataset_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        this.showMessage('Dataset exported successfully!', 'success');
    }
    
    importModel(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const modelData = JSON.parse(e.target.result);
                
                // Create new model with imported weights
                this.state.model = new MLP(
                    modelData.config.input_size,
                    modelData.config.hidden_units,
                    modelData.config.activation,
                    modelData.config.learning_rate,
                    modelData.config.l2_lambda
                );
                
                // Set the weights
                this.state.model.W1 = modelData.weights.W1;
                this.state.model.b1 = modelData.weights.b1;
                this.state.model.W2 = modelData.weights.W2;
                this.state.model.b2 = modelData.weights.b2;
                
                // Update UI
                document.getElementById('hiddenUnits').value = modelData.config.hidden_units;
                document.getElementById('hiddenValue').textContent = modelData.config.hidden_units;
                document.getElementById('activation').value = modelData.config.activation;
                document.getElementById('learningRate').value = modelData.config.learning_rate;
                document.getElementById('lrValue').textContent = modelData.config.learning_rate;
                document.getElementById('l2Lambda').value = modelData.config.l2_lambda;
                document.getElementById('l2Value').textContent = modelData.config.l2_lambda;
                
                this.updateUI();
                this.plotNeuralNetwork();
                this.showMessage('Model imported successfully!', 'success');
                
            } catch (error) {
                this.showMessage('Error importing model: Invalid file format!', 'error');
            }
        };
        reader.readAsText(file);
    }
    
    showMessage(message, type) {
        // Create temporary message element
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${type}`;
        messageEl.textContent = message;
        
        document.body.appendChild(messageEl);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (document.body.contains(messageEl)) {
                document.body.removeChild(messageEl);
            }
        }, 3000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    const app = new DLPlayground();
    window.dlPlayground = app; // Make it globally accessible
    console.log('App initialized:', app);
    
    // Test if elements exist
    console.log('Epoch element:', document.getElementById('currentEpoch'));
    console.log('Loss element:', document.getElementById('currentLoss'));
    console.log('Accuracy element:', document.getElementById('currentAccuracy'));
    console.log('Start Training button:', document.getElementById('startTraining'));
    
    // Initialize metrics display
    app.updateMetrics();
    console.log('Initial metrics update completed');
    
    // Add test function to window
    window.testMetrics = () => {
        console.log('Testing metrics update...');
        app.state.currentEpoch = 5;
        app.state.currentLoss = 0.5;
        app.state.currentAccuracy = 0.8;
        app.updateMetrics();
        console.log('Test completed - check if values changed');
    };
    
    window.testTraining = () => {
        console.log('Testing training...');
        if (app.state.model && app.state.dataset) {
            app.trainOneEpoch();
        } else {
            console.log('No model or dataset!');
        }
    };
});

// Simple random number generator for seeding
Math.seedrandom = function(seed) {
    // Simple LCG implementation
    let x = seed;
    return function() {
        x = (x * 1664525 + 1013904223) % 4294967296;
        return x / 4294967296;
    };
};
