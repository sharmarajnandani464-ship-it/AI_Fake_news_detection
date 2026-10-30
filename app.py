<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fake News Detection</title>
</head>
<body>
    <h1>AI Fake News Detection</h1>

    <form action="/" method="POST">
        <textarea name="news" placeholder="Enter news here..." required></textarea>
        <br><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
        <h2>Result: {{ prediction }}</h2>
    {% endif %}

    <a href="/history">History</a>
    <br>
    <a href="/about">About</a>
</body>
</html>
