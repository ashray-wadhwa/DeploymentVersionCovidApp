var express = require('express');
var router = express.Router();
const mongoose = require('mongoose');
const myModel = require('./schema');


const dbRoute =
    'mongodb+srv://dbUser:malloo4301@myCluster-puppf.mongodb.net/donationInfo?retryWrites=true&w=majority';//*
mongoose.connect(dbRoute,{ useNewUrlParser: true, useUnifiedTopology: true})
    .then(() => console.log("Database Connected Successfully"))
    .catch(err => console.log(err));

let db = mongoose.connection;//*

db.once('open', () => console.log('connected to the database'));//*

// checks if connection with the database is successful
db.on('error', console.error.bind(console, 'MongoDB connection error:'));//*


router.get('/', function (req, res, next) {
    console.log("Entered get");
    myModel.find(function(err,data){
        if(err){
            return res.json({success: false, error:err});
        }else{
            return res.json({success:true, info: data})
        }
    });
    
});

router.post('/', function (req, res, next) {
    console.log("Entered post");
    let po = new myModel();
    po.Organization = req.body.Organization;
    po.Description = req.body.Description;
    po.Links = req.body.Links;
    po.save((err) => {
        if (err) return res.json({ success: false, error: err });
        return res.json({ success: true });
    });
});


//Delete Working and checking only for country
// router.delete('/', function (req, res, next) {
//     console.log("Entered delete");
//     myModel.findOneAndRemove({Organization: req.body.Organization}, (err)=>{
//     if (err) 
//         return res.json({ success: false, error: err });
//     else {
//         return res.json({ success: true });
//     }
//     });
//  });

//  router.put('/', function (req, res, next){
//     console.log("Entered put");
//     let po = new myModel();
//     po.Organization = req.body.Organization;
//     po.Description = req.body.Description;
//     po.Links = req.body.Links;

//     myModel.findOneAndRemove({Organization: req.body.Organization}, (err)=>{
//         if (err) 
//             return res.json({ success: false, error: err });
//         else {
//             po.save((err) => {
//                 if (err) return res.json({ success: false, error: err });
//                 return res.json({ success: true });
//             });
//         }
//         });
// });

module.exports = router;