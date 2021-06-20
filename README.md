# Movie Night Group Recommendation

<div align="center">
  <p>
  <img width="230" alt="Group_movie" src="https://user-images.githubusercontent.com/60047427/122675863-a7e86380-d1db-11eb-84f4-d4a3bc488209.jpg">

  </p>
  <p>
    <a href="">
      <img alt="First release" src="https://img.shields.io/badge/release-v1.0-brightgreen.svg" />
    </a>
  </p>
</div>

## Website
Visit: [https://movie-night-recommendation.herokuapp.com](https://movie-night-recommendation.herokuapp.com) <br>
`Please note that it might take some time for the Heroku application to load, so please be patient.`

## Project setup
1. `git clone https://github.com/GroupMovieRec/group-movie-recommendation.git`
2. `cd group-movie-recommendation`
3. `python app.py`
4. Open browser at the respective address. **(Chrome or Firefox recommended)** <br> For example, that address might look something like: `http://0.0.0.0:5000/`.

## Testing setup
In the `testing/` folder you'll be able to find all necessary files in order to run the tests for comparing our method's performance.
- `testing_group_differences.py` and `testing_method_comparison.py` import `nmf.py`, `rdfnmf.py`, and `rdfnmf_updated.py`.
- `nmf.py` is by created by Hung-Hsuan Chen <hhchen1105@gmail.com>.
- `rdfnmf.py` and `rdfnmf_updated.py` are heavily modified by us, but are based on Hung-Hsuan Chen's work.
- In the testing files you can change the 50 iterations to something less. When you create the models by initialising and declaring a class for the different methods you can also reset the n_epochs to something less than 50.

## Acknowledgements
- We would like to thank the authors of Movinder for publishing the code for their application, as a part of that has been the baseline for our user-interface. If you’re curious about the [Movinder](movinder.herokuapp.com/) project, make sure to check them out their [GitHub repository](https://github.com/Movinder).
- We would also like the thank Chen & Chen for publishing their code for the RDFNMF, as this was the baseline for our weighted matrix factorisation implementation. If you are interest in there work, check out their [GitHub repository](https://github.com/ncu-dart/rdf) for their [paper](https://dl.acm.org/doi/10.1145/3285954). 

## Information
Group project created in the context of TU Delft's CS4065 Multimedia Search and Recommendation.

Team 8:
- Shreyan Biswas
- Caroline Freyer
- Francesca Drummer
- Stefan Petrescu

Video presentation can be watched [here](https://www.youtube.com/watch?v=twg5SDrTw3U).
