from flask import Flask
# from flask import jsonify
# from dev_global.basic import deamon


# Usage: import app, and app.run()
app = Flask('Proton')


# On Server
@app.route('/test')
def test():
    return "Hello world!"


if __name__ == "__main__":
    app.run()
