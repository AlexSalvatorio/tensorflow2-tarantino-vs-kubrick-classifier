var Scraper = require ('images-scraper');
 
let google = new Scraper.Google({
    keyword: 'For few dollar more'+' screen capture',
    limit: 200,
    puppeteer: {
        headless: false
    },
  tbs: {
        // every possible tbs search option, some examples and more info: http://jwebnet.net/advancedgooglesearch.html
    isz: undefined, 				// options: l(arge), m(edium), i(cons), etc. 
    itp: undefined, 				// options: clipart, face, lineart, news, photo
        ic: undefined, 					// options: color, gray, trans
        sur: undefined,					// options: fmc (commercial reuse with modification), fc (commercial reuse), fm (noncommercial reuse with modification), f (noncommercial reuse)
  }
});
 
(async () => {
    const results = await google.start();
    console.log('results',results);
})();

/*******************
 * UTILS
 *******************/

function download (uri, filename, callback){
    request.head(uri, function(err, res, body){
        if(res != undefined){
            console.log('content-type:', res.headers['content-type']);
            console.log('content-length:', res.headers['content-length']);
            request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
        } else {
            console.error("URI not defined: "+uri);
        }
    });
  };