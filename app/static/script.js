async function predictName() {

    const nameInput = document.getElementById("name");

    const resultBox = document.getElementById("result");

    const name = nameInput.value.trim();

    if (!name) {

        alert("Please enter a name.");

        return;
    }

    // Show loading

    resultBox.innerHTML = `
        <div class="loading">
            Analyzing "${name}"...
        </div>
    `;

    try {

        console.log("Sending prediction request...");
        console.log("Name:", name);

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    name: name
                })
            }
        );

        console.log(
            "HTTP status:",
            response.status
        );

        const data = await response.json();

        console.log(
            "Server response:",
            data
        );

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction request failed."
            );
        }

        if (
            !data.success ||
            !data.prediction
        ) {

            console.error(
                "Unexpected response:",
                data
            );

            throw new Error(
                "Server returned an unexpected prediction format."
            );
        }

        const maleProbability =
            Number(
                data.male_probability
            );

        const femaleProbability =
            Number(
                data.female_probability
            );

        const confidence =
            Number(
                data.confidence
            );

        // Check numbers

        if (
            !Number.isFinite(
                maleProbability
            ) ||
            !Number.isFinite(
                femaleProbability
            ) ||
            !Number.isFinite(
                confidence
            )
        ) {

            console.error(
                "Invalid numeric values:",
                data
            );

            throw new Error(
                "Invalid probability values received from server."
            );
        }

        resultBox.innerHTML = `

            <div class="prediction-result">

                <h2>
                    ${escapeHtml(data.prediction)}
                </h2>

                <p>
                    Name:
                    <strong>
                        ${escapeHtml(data.name)}
                    </strong>
                </p>

                <div class="probability-row">

                    <div>
                        <span>Male</span>

                        <strong>
                            ${maleProbability.toFixed(2)}%
                        </strong>
                    </div>

                    <div>
                        <span>Female</span>

                        <strong>
                            ${femaleProbability.toFixed(2)}%
                        </strong>
                    </div>

                </div>

                <div class="confidence">

                    Confidence:
                    <strong>
                        ${confidence.toFixed(2)}%
                    </strong>

                    <span>
                        (${escapeHtml(
                            data.confidence_level
                        )})
                    </span>

                </div>

                <div class="model-details">

                    <p>
                        Baseline Bi-LSTM:
                        ${Number(
                            data.baseline_male_probability
                        ).toFixed(2)}% male
                    </p>

                    <p>
                        Regularized Bi-LSTM:
                        ${Number(
                            data.regularized_male_probability
                        ).toFixed(2)}% male
                    </p>

                    <p>
                        Final model:
                        50/50 Bi-LSTM Ensemble
                    </p>

                </div>

            </div>
        `;

    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        resultBox.innerHTML = `

            <div class="error">

                <strong>
                    Prediction failed
                </strong>

                <p>
                    ${escapeHtml(
                        error.message
                    )}
                </p>

            </div>

        `;
    }
}

document
    .getElementById("name")
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
            ) {

                event.preventDefault();

                predictName();
            }

        }
    );

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value);

    return div.innerHTML;
}